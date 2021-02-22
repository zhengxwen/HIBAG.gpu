// ===============================================================
//
// HIBAG.gpu R package (GPU-based implementation for the HIBAG package)
// Copyright (C) 2017-2021    Xiuwen Zheng (zhengx@u.washington.edu)
// All rights reserved.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.	 If not, see <http://www.gnu.org/licenses/>.


// Optimization level
#ifndef HIBAG_NO_COMPILER_OPTIMIZE
#if defined(__clang__) && !defined(__APPLE__)
    #pragma clang optimize on
#endif
#if defined(__GNUC__) && ((__GNUC__>4) || (__GNUC__==4 && __GNUC_MINOR__>=4))
    #pragma GCC optimize("O3")
    #define MATH_OFAST    __attribute__((optimize("Ofast")))
#endif
#endif

#ifndef MATH_OFAST
#   define MATH_OFAST
#endif


// Defined in HIBAG/install/LibHLA_ext.h
#define HIBAG_STRUCTURE_HEAD_ONLY
#include <LibHLA_ext.h>

#include <string.h>
#include <cstdlib>
#include <cmath>
#include <vector>

#ifdef __APPLE__
#   define CL_SILENCE_DEPRECATION
#	include <OpenCL/opencl.h>
#else
#	include <CL/opencl.h>
#endif

#define USE_RINTERNALS 1
#include <Rinternals.h>
#include <Rdefines.h>


// disable timing
// #define HIBAG_ENABLE_TIMING

#ifdef HIBAG_ENABLE_TIMING
#   include <time.h>
#endif


#define GPU_CREATE_MEM(x, flag, size, ptr)	  \
	x = clCreateBuffer(gpu_context, flag, size, ptr, &err); \
	if (!x) throw err_text("Unable to create buffer " #x, err);

#define GPU_FREE_MEM(x)	   if (x) { \
		cl_int err = clReleaseMemObject(x); \
		if (err != CL_SUCCESS) \
			throw err_text("Failed to free memory buffer " #x, err); \
		x = NULL; \
	}

#define GPU_READ_MEM(x, size, ptr)	  \
	err = clEnqueueReadBuffer(gpu_command_queue, x, CL_TRUE, 0, size, ptr, 0, NULL, NULL); \
	if (err != CL_SUCCESS) \
		throw err_text("Failed to read memory buffer " #x, err);

#define GPU_WRITE_MEM(x, offset, size, ptr)	   \
	err = clEnqueueWriteBuffer(gpu_command_queue, x, CL_TRUE, offset, size, ptr, 0, NULL, NULL); \
	if (err != CL_SUCCESS) \
		throw err_text("Failed to write memory buffer " #x, err);

#define GPU_SETARG(kernel, i, x)	\
	err = clSetKernelArg(kernel, i, sizeof(x), &x); \
	if (err != CL_SUCCESS) \
		throw err_text("Failed to set kernel (" #kernel ") argument (" #i ")", err);

#define GPU_RUN_KERNAL(kernel, ndim, wdims, lsize)	  \
	{ \
	cl_event e; \
	err = clEnqueueNDRangeKernel(gpu_command_queue, kernel, ndim, NULL, \
		wdims, lsize, 0, NULL, &e); \
	if (err != CL_SUCCESS) \
		throw err_text("Failed to run clEnqueueNDRangeKernel() with " #kernel, err); \
	err = clWaitForEvents(1, &e); \
	if (err != CL_SUCCESS) \
		throw err_text("Failed to run clWaitForEvents() with " #kernel, err); \
	}


namespace HLA_LIB
{
	using namespace std;

	// Packed bi-allelic SNP haplotype structure: 8 alleles in a byte
	// struct THaplotype, defined LibHLA_ext.h

	// An unordered pair of HLA alleles
	// struct THLAType, defined LibHLA_ext.h

	// Packed bi-allelic SNP genotype structure: 8 SNPs in a byte
	// struct TGenotype, defined LibHLA_ext.h

	// Pointer to the structure of functions using GPU
	// TypeGPUExtProc defined in LibHLA_ext.h
	struct TypeGPUExtProc GPU_Proc;

	// HIBAG.gpu:::.packageEnv
	SEXP packageEnv = NULL;

	// the number of unique HLA alleles
	static int Num_HLA;
	// the number of samples
	static int Num_Sample;
	// the number of individual classifiers
	static int Num_Classifier;


	// flags for usage of double or single precision
	static bool gpu_f64_build_flag = false;
	static bool gpu_f64_pred_flag  = false;

	// OpenCL device variables
	static cl_context gpu_context = NULL;
	static cl_command_queue gpu_command_queue = NULL;

	// OpenCL kernel functions
	static cl_kernel gpu_kl_build_calc_prob	 = NULL;
	static cl_kernel gpu_kl_build_find_maxprob = NULL;
	static cl_kernel gpu_kl_build_sum_prob = NULL;
	static cl_kernel gpu_kl_pred_calc = NULL;
	static cl_kernel gpu_kl_pred_sumprob = NULL;
	static cl_kernel gpu_kl_pred_addprob = NULL;
	static cl_kernel gpu_kl_clear_mem = NULL;

	// OpenCL memory objects

	// parameters, int[] =
	//   [ # of haplotypes, # of SNPs, starting sample index, offset, ... ]
	static cl_mem mem_build_param = NULL;
	// parameter offset
	static const int offset_build_param = 4;

	// SNP genotypes, TGenotype[]
	static cl_mem mem_snpgeno = NULL;

	// haplotype list, THaplotype[]
	static cl_mem mem_haplo_list = NULL;

	// the buffer of probabilities
	// double[nHLA*(nHLA+1)/2][# of samples] -- prob. for classifiers or samples
	static cl_mem mem_prob_buffer = NULL;
	// sizeof(double[nHLA*(nHLA+1)/2]), or sizeof(float[nHLA*(nHLA+1)/2])
	static size_t msize_prob_buffer = 0;
	// build:   msize_prob_buffer_total = msize_prob_buffer * num_sample
	// predict: msize_prob_buffer_total = msize_prob_buffer * num_classifier
	static size_t msize_prob_buffer_total = 0;

	// max. index or sum of prob.
	static cl_mem mem_build_output = NULL;

	// the max number of samples can be hold in mem_prob_buffer
	static int mem_sample_nmax = 0;
	// the max number of haplotypes can be hold in mem_haplo_list
	static int build_haplo_nmax = 0;

	// num of haplotypes and SNPs for each classifier: int[][2]
	static cl_mem mem_pred_haplo_num = NULL;

	// classifier weight
	static cl_mem mem_pred_weight = NULL;

	static int wdim_pred_addprob = 0;


	// used for work-group size (1-dim and 2-dim)
	const  size_t gpu_const_local_size = 64;
	static size_t gpu_local_size_d1 = 64;  // will be determined automatically
	static size_t gpu_local_size_d2 = 8;   // will be determined automatically



	// ===================================================================== //
	// building classifiers

	static int build_num_oob;  ///< the number of out-of-bag samples
	static int build_num_ib;   ///< the number of in-bag samples
	static int run_num_haplo;  ///< the total number of haplotypes
	static int run_num_snp;    ///< the number of SNPs
	static int wdim_num_haplo; ///< global_work_size for the number of haplotypes

	static vector<int> hla_map_index;


	// ===================================================================== //

	// return OpenCL error code
	static const char *err_code(int err)
	{
		#define ERR_RET(s)    case s: return #s;
		switch (err)
		{
			ERR_RET(CL_SUCCESS)
			ERR_RET(CL_DEVICE_NOT_FOUND)
			ERR_RET(CL_DEVICE_NOT_AVAILABLE)
			ERR_RET(CL_COMPILER_NOT_AVAILABLE)
			ERR_RET(CL_MEM_OBJECT_ALLOCATION_FAILURE)
			ERR_RET(CL_OUT_OF_RESOURCES)
			ERR_RET(CL_OUT_OF_HOST_MEMORY)
			ERR_RET(CL_PROFILING_INFO_NOT_AVAILABLE)
			ERR_RET(CL_MEM_COPY_OVERLAP)
			ERR_RET(CL_IMAGE_FORMAT_MISMATCH)
			ERR_RET(CL_IMAGE_FORMAT_NOT_SUPPORTED)
			ERR_RET(CL_BUILD_PROGRAM_FAILURE)
			ERR_RET(CL_MAP_FAILURE)
			ERR_RET(CL_INVALID_VALUE)
			ERR_RET(CL_INVALID_DEVICE_TYPE)
			ERR_RET(CL_INVALID_PLATFORM)
			ERR_RET(CL_INVALID_DEVICE)
			ERR_RET(CL_INVALID_CONTEXT)
			ERR_RET(CL_INVALID_QUEUE_PROPERTIES)
			ERR_RET(CL_INVALID_COMMAND_QUEUE)
			ERR_RET(CL_INVALID_HOST_PTR)
			ERR_RET(CL_INVALID_MEM_OBJECT)
			ERR_RET(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR)
			ERR_RET(CL_INVALID_IMAGE_SIZE)
			ERR_RET(CL_INVALID_SAMPLER)
			ERR_RET(CL_INVALID_BINARY)
			ERR_RET(CL_INVALID_BUILD_OPTIONS)
			ERR_RET(CL_INVALID_PROGRAM)
			ERR_RET(CL_INVALID_PROGRAM_EXECUTABLE)
			ERR_RET(CL_INVALID_KERNEL_NAME)
			ERR_RET(CL_INVALID_KERNEL_DEFINITION)
			ERR_RET(CL_INVALID_KERNEL)
			ERR_RET(CL_INVALID_ARG_INDEX)
			ERR_RET(CL_INVALID_ARG_VALUE)
			ERR_RET(CL_INVALID_ARG_SIZE)
			ERR_RET(CL_INVALID_KERNEL_ARGS)
			ERR_RET(CL_INVALID_WORK_DIMENSION)
			ERR_RET(CL_INVALID_WORK_GROUP_SIZE)
			ERR_RET(CL_INVALID_WORK_ITEM_SIZE)
			ERR_RET(CL_INVALID_GLOBAL_OFFSET)
			ERR_RET(CL_INVALID_EVENT_WAIT_LIST)
			ERR_RET(CL_INVALID_EVENT)
			ERR_RET(CL_INVALID_OPERATION)
			ERR_RET(CL_INVALID_GL_OBJECT)
			ERR_RET(CL_INVALID_BUFFER_SIZE)
			ERR_RET(CL_INVALID_MIP_LEVEL)
			ERR_RET(CL_INVALID_GLOBAL_WORK_SIZE)

		#ifdef CL_VERSION_1_1
			ERR_RET(CL_MISALIGNED_SUB_BUFFER_OFFSET)
			ERR_RET(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST)
			ERR_RET(CL_INVALID_PROPERTY)
		#endif
		#ifdef CL_VERSION_1_2
			ERR_RET(CL_COMPILE_PROGRAM_FAILURE)
			ERR_RET(CL_LINKER_NOT_AVAILABLE)
			ERR_RET(CL_LINK_PROGRAM_FAILURE)
			ERR_RET(CL_DEVICE_PARTITION_FAILED)
			ERR_RET(CL_KERNEL_ARG_INFO_NOT_AVAILABLE)
			ERR_RET(CL_INVALID_IMAGE_DESCRIPTOR)
			ERR_RET(CL_INVALID_COMPILER_OPTIONS)
			ERR_RET(CL_INVALID_LINKER_OPTIONS)
			ERR_RET(CL_INVALID_DEVICE_PARTITION_COUNT)
		#endif
		#ifdef CL_VERSION_2_0
			ERR_RET(CL_INVALID_PIPE_SIZE)
			ERR_RET(CL_INVALID_DEVICE_QUEUE)
		#endif
		#ifdef CL_VERSION_2_2
			ERR_RET(CL_INVALID_SPEC_ID)
			ERR_RET(CL_MAX_SIZE_RESTRICTION_EXCEEDED)
		#endif
		}
		return "Unknown";
		#undef ERR_RET
	}

	// OpenCL error message
	static const char *err_text(const char *txt, int err)
	{
		static char buf[1024];
		sprintf(buf, "%s (error: %d, %s).", txt, err, err_code(err));
		return buf;
	}


	// Map and unmap GPU memory buffer
	template<typename TYPE> struct GPU_MEM_MAP
	{
	public:
		GPU_MEM_MAP(cl_mem mem, size_t size, bool readonly)
		{
			cl_int err;
			gpu_mem = mem;
			void *p = clEnqueueMapBuffer(gpu_command_queue, gpu_mem, CL_TRUE,
				CL_MAP_READ | (readonly ? 0 : CL_MAP_WRITE),
				0, size * sizeof(TYPE), 0, NULL, NULL, &err);
			mem_ptr = (TYPE*)p;
			if (!p)
				throw err_text("Unable to map buffer to host memory", err);
		}
		~GPU_MEM_MAP()
		{
			if (mem_ptr)
			{
				clEnqueueUnmapMemObject(gpu_command_queue, gpu_mem, (void*)mem_ptr,
					0, NULL, NULL);
				mem_ptr = NULL;
			}
		}
		inline TYPE *ptr() { return mem_ptr; }
		inline const TYPE *ptr() const { return mem_ptr; }
	private:
		cl_mem gpu_mem;
		TYPE *mem_ptr;
	};

}

using namespace std;
using namespace HLA_LIB;


// ========================================================================= //

#ifdef HIBAG_ENABLE_TIMING

static clock_t timing_array[7];

template<size_t I> struct TTiming
{
	clock_t start;
	inline TTiming() { start = clock(); }
	inline ~TTiming() { Stop(); }
	inline void Stop() { timing_array[I] += clock() - start; }
};

#define HIBAG_TIMING(i)    TTiming<i> tm;
#define TM_BUILD_TOTAL         0
#define TM_BUILD_OOB_CLEAR     1
#define TM_BUILD_OOB_PROB      2
#define TM_BUILD_OOB_MAXIDX    3
#define TM_BUILD_IB_CLEAR      4
#define TM_BUILD_IB_PROB       5
#define TM_BUILD_IB_LOGLIK     6

#else
#   define HIBAG_TIMING(i)
#endif


// ========================================================================= //

extern "C"
{

/// get the external R object
inline static SEXP get_var_env(const char *varnm)
{
	SEXP rv_ans = Rf_findVarInFrame(packageEnv, Rf_install(varnm));
	if (rv_ans == R_NilValue)
		Rf_error("No '%s' in .packageEnv.", varnm);
	return rv_ans;
}

/// get the OpenCL device
inline static cl_device_id get_device_env(const char *varnm)
{
	SEXP dev = get_var_env(varnm);
	if (!Rf_inherits(dev, "clDeviceID") || TYPEOF(dev) != EXTPTRSXP)
		Rf_error("'.packageEnv$%s' is not an OpenCL device.", varnm);
	return (cl_device_id)R_ExternalPtrAddr(dev);
}

/// get the OpenCL context
inline static cl_context get_context_env(const char *varnm)
{
	SEXP ctx = get_var_env(varnm);
	if (!Rf_inherits(ctx, "clContext") || TYPEOF(ctx) != EXTPTRSXP)
		Rf_error("'.packageEnv$%s' is not an OpenCL context.", varnm);
	return (cl_context)R_ExternalPtrAddr(ctx);
}

/// get the OpenCL command queue
inline static cl_command_queue get_command_queue_env(const char *varnm)
{
	SEXP ctx = get_var_env(varnm);
	if (!Rf_inherits(ctx, "clContext") || TYPEOF(ctx) != EXTPTRSXP)
		Rf_error("'.packageEnv$%s' is not an OpenCL context.", varnm);
	SEXP queue = Rf_getAttrib(ctx, Rf_install("queue"));
	if (!Rf_inherits(queue, "clCommandQueue") || TYPEOF(queue) != EXTPTRSXP)
		Rf_error("Expected OpenCL command queue");
	return (cl_command_queue)R_ExternalPtrAddr(queue);
}

/// get the OpenCL kernel
inline static cl_kernel get_kernel_env(const char *varnm)
{
	SEXP k = get_var_env(varnm);
	if (!Rf_inherits(k, "clKernel") || TYPEOF(k) != EXTPTRSXP)
		Rf_error("'.packageEnv$%s' is not an OpenCL kernel.", varnm);
	return (cl_kernel)R_ExternalPtrAddr(k);
}

/// get the OpenCL kernel
inline static cl_mem get_mem_env(const char *varnm)
{
	SEXP m = get_var_env(varnm);
	if (!Rf_inherits(m, "clBuffer") || TYPEOF(m) != EXTPTRSXP)
		Rf_error("'.packageEnv$%s' is not an OpenCL buffer.", varnm);
	return (cl_mem)R_ExternalPtrAddr(m);
}

static int get_kernel_param(cl_device_id dev, cl_kernel kernel,
	cl_kernel_work_group_info param)
{
	size_t n = 0;
	cl_int err = clGetKernelWorkGroupInfo(kernel, dev, param, sizeof(n), &n, NULL);
	if (err != CL_SUCCESS) return NA_INTEGER;
	return n;
}

/// clear the memory buffer 'mem_prob_buffer'
static inline void clear_prob_buffer(size_t size)
{
	cl_int err;
#if defined(CL_VERSION_1_2) && 0
	int zero = 0;
	err = clEnqueueFillBuffer(gpu_command_queue, mem_prob_buffer,
		&zero, sizeof(zero), 0, size, 0, NULL, NULL);
	if (err != CL_SUCCESS)
		throw err_text("clEnqueueFillBuffer() with mem_prob_buffer failed", err);
#else
	if (size >= 4294967296)
		throw "size is too large in clear_prob_buffer().";
	int n = size / 4;
	GPU_SETARG(gpu_kl_clear_mem, 0, n);
	size_t wdim = n / gpu_local_size_d1;
	if (n % gpu_local_size_d1) wdim++;
	wdim *= gpu_local_size_d1;
	GPU_RUN_KERNAL(gpu_kl_clear_mem, 1, &wdim, &gpu_local_size_d1);
#endif
}


/// get the sum of double array
inline static MATH_OFAST double get_sum_f64(const double p[], size_t n)
{
	double sum = 0;
	for (size_t i=0; i < n; i++) sum += p[i];
	return sum;
}

/// mul operation
inline static MATH_OFAST void fmul_f64(double p[], size_t n, double scalar)
{
	for (size_t i=0; i < n; i++) p[i] *= scalar;
}


// ===================================================================== //

static bool gpu_verbose = false;

/// release the memory buffer object
SEXP gpu_set_verbose(SEXP verbose)
{
	gpu_verbose = (Rf_asLogical(verbose)==TRUE);
	return R_NilValue;
}


// ===================================================================== //

// initialize the internal structure for building a model
void build_init(int nHLA, int nSample)
{
#ifdef HIBAG_ENABLE_TIMING
	memset(timing_array, 0, sizeof(timing_array));
	timing_array[TM_BUILD_TOTAL] = clock();
#endif

	// initialize
	Num_HLA = nHLA;
	Num_Sample = nSample;
	const size_t size_hla = nHLA * (nHLA+1) >> 1;
	hla_map_index.resize(size_hla*2);
	int *p = &hla_map_index[0];
	for (int h1=0; h1 < nHLA; h1++)
		for (int h2=h1; h2 < nHLA; h2++, p+=2) { p[0] = h1; p[1] = h2; }

	// 64-bit floating-point number or not?
	gpu_f64_build_flag = Rf_asLogical(get_var_env("flag_build_f64")) == TRUE;

	// device variables
	cl_int err;
	gpu_context = get_context_env("gpu_context");
	gpu_command_queue = get_command_queue_env("gpu_context");

	// kernels
	gpu_kl_build_calc_prob    = get_kernel_env("kernel_build_calc_prob");
	gpu_kl_build_find_maxprob = get_kernel_env("kernel_build_find_maxprob");
	gpu_kl_build_sum_prob     = get_kernel_env("kernel_build_sum_prob");
	gpu_kl_clear_mem          = get_kernel_env("kernel_clear_mem");

	// GPU memory
	cl_mem mem_rare_freq = get_mem_env(gpu_f64_build_flag ?
		"mem_exp_log_min_rare_freq64" : "mem_exp_log_min_rare_freq32");
	mem_prob_buffer = get_mem_env("mem_prob_buffer");
	msize_prob_buffer = nHLA*(nHLA+1)/2 * (gpu_f64_build_flag ? 64 : 32);
	mem_build_param = get_mem_env("mem_build_param");
	mem_haplo_list = get_mem_env("mem_haplo_list");
	mem_snpgeno = get_mem_env("mem_snpgeno");
	build_haplo_nmax = Rf_asInteger(get_var_env("build_haplo_nmax"));
	mem_sample_nmax = Rf_asInteger(get_var_env("build_sample_nmax"));
	msize_prob_buffer_total = msize_prob_buffer * mem_sample_nmax;
	mem_build_output = get_mem_env("mem_build_output");

	// arguments for build_calc_prob
	int sz_hla = size_hla;
	GPU_SETARG(gpu_kl_build_calc_prob, 0, mem_prob_buffer);
	GPU_SETARG(gpu_kl_build_calc_prob, 1, nHLA);
	GPU_SETARG(gpu_kl_build_calc_prob, 2, sz_hla);
	GPU_SETARG(gpu_kl_build_calc_prob, 3, mem_rare_freq);
	GPU_SETARG(gpu_kl_build_calc_prob, 4, mem_build_param);
	GPU_SETARG(gpu_kl_build_calc_prob, 5, mem_haplo_list);
	GPU_SETARG(gpu_kl_build_calc_prob, 6, mem_snpgeno);

	// arguments for gpu_kl_build_find_maxprob (out-of-bag)
	GPU_SETARG(gpu_kl_build_find_maxprob, 0, mem_build_output);
	GPU_SETARG(gpu_kl_build_find_maxprob, 1, sz_hla);
	GPU_SETARG(gpu_kl_build_find_maxprob, 2, mem_prob_buffer);

	// arguments for gpu_kl_build_sum_prob (in-bag)
	GPU_SETARG(gpu_kl_build_sum_prob, 0, mem_build_output);
	GPU_SETARG(gpu_kl_build_sum_prob, 1, nHLA);
	GPU_SETARG(gpu_kl_build_sum_prob, 2, sz_hla);
	GPU_SETARG(gpu_kl_build_sum_prob, 3, mem_build_param);
	GPU_SETARG(gpu_kl_build_sum_prob, 4, mem_snpgeno);
	GPU_SETARG(gpu_kl_build_sum_prob, 5, mem_prob_buffer);

	// arguments for gpu_kl_build_clear_mem
	GPU_SETARG(gpu_kl_clear_mem, 1, mem_prob_buffer);
}


void build_done()
{
	gpu_kl_build_calc_prob =
		gpu_kl_build_find_maxprob =
		gpu_kl_build_sum_prob =
		gpu_kl_clear_mem = NULL;
	mem_build_param =
		mem_snpgeno =
		mem_build_output =
		mem_haplo_list =
		mem_prob_buffer = NULL;
	hla_map_index.clear();

#ifdef HIBAG_ENABLE_TIMING
	timing_array[TM_BUILD_TOTAL] = clock() - timing_array[TM_BUILD_TOTAL];
	Rprintf("GPU implementation took %0.2f seconds in total:\n"
			"    OOB init(): %0.2f%%, %0.2fs\n"
			"    OOB prob(): %0.2f%%, %0.2fs\n"
			"    OOB max index: %0.2f%%, %0.2fs\n"
			"    IB init(): %0.2f%%, %0.2fs\n"
			"    IB prob(): %0.2f%%, %0.2fs\n"
			"    IB log likelihood(): %0.2f%%, %0.2fs\n",
		((double)timing_array[TM_BUILD_TOTAL]) / CLOCKS_PER_SEC,
		100.0 * timing_array[TM_BUILD_OOB_CLEAR] / timing_array[TM_BUILD_TOTAL],
		((double)timing_array[TM_BUILD_OOB_CLEAR]) / CLOCKS_PER_SEC,
		100.0 * timing_array[TM_BUILD_OOB_PROB] / timing_array[TM_BUILD_TOTAL],
		((double)timing_array[TM_BUILD_OOB_PROB]) / CLOCKS_PER_SEC,
		100.0 * timing_array[TM_BUILD_OOB_MAXIDX] / timing_array[TM_BUILD_TOTAL],
		((double)timing_array[TM_BUILD_OOB_MAXIDX]) / CLOCKS_PER_SEC,
		100.0 * timing_array[TM_BUILD_IB_CLEAR] / timing_array[TM_BUILD_TOTAL],
		((double)timing_array[TM_BUILD_IB_CLEAR]) / CLOCKS_PER_SEC,
		100.0 * timing_array[TM_BUILD_IB_PROB] / timing_array[TM_BUILD_TOTAL],
		((double)timing_array[TM_BUILD_IB_PROB]) / CLOCKS_PER_SEC,
		100.0 * timing_array[TM_BUILD_IB_LOGLIK] / timing_array[TM_BUILD_TOTAL],
		((double)timing_array[TM_BUILD_IB_LOGLIK]) / CLOCKS_PER_SEC
	);
#endif
}

void build_set_bootstrap(const int oob_cnt[])
{
	GPU_MEM_MAP<int> M(mem_build_param, offset_build_param + 2*Num_Sample, false);
	int *p_oob = M.ptr() + offset_build_param;
	int *p_ib  = p_oob + Num_Sample;
	build_num_oob = build_num_ib  = 0;
	for (int i=0; i < Num_Sample; i++)
	{
		if (oob_cnt[i] <= 0)
			p_oob[build_num_oob++] = i;
		else
			p_ib[build_num_ib++] = i;
	}
}

void build_set_haplo_geno(const THaplotype haplo[], int n_haplo,
	const TGenotype geno[], int n_snp)
{
	if (n_haplo > build_haplo_nmax)
		throw "Too many haplotypes out of the limit, please contact the package author.";

	cl_int err;
	wdim_num_haplo = run_num_haplo = n_haplo;
	if (wdim_num_haplo % gpu_local_size_d2)
		wdim_num_haplo = (wdim_num_haplo/gpu_local_size_d2 + 1)*gpu_local_size_d2;
	GPU_WRITE_MEM(mem_haplo_list, 0, sizeof(THaplotype)*n_haplo, (void*)haplo);

	run_num_snp = n_snp;
	GPU_WRITE_MEM(mem_snpgeno, 0, sizeof(TGenotype)*Num_Sample, (void*)geno);
}

inline static int compare(const THLAType &H1, const THLAType &H2)
{
	int P1=H1.Allele1, P2=H1.Allele2;
	int T1=H2.Allele1, T2=H2.Allele2;
	int cnt = 0;
	if ((P1==T1) || (P1==T2))
	{
		cnt = 1;
		if (P1==T1) T1 = -1; else T2 = -1;
	}
	if ((P2==T1) || (P2==T2)) cnt ++;
	return cnt;
}

int build_acc_oob()
{
	if (build_num_oob <= 0) return 0;
	if (build_num_oob > mem_sample_nmax)
		throw "Too many sample out of the limit of GPU memory, please contact the package author.";

	cl_int err;
	// initialize
	{
		HIBAG_TIMING(TM_BUILD_OOB_CLEAR)
		clear_prob_buffer(msize_prob_buffer * build_num_oob);
		int param[4] = { run_num_haplo, run_num_snp, 0, offset_build_param };
		GPU_WRITE_MEM(mem_build_param, 0, sizeof(param), param);
	}

	// run OpenCL (calculating probabilities)
	{
		HIBAG_TIMING(TM_BUILD_OOB_PROB)
		size_t wdims[3] =
			{ (size_t)wdim_num_haplo, (size_t)wdim_num_haplo, (size_t)build_num_oob };
		size_t local_size[3] =
			{ gpu_local_size_d2, gpu_local_size_d2, 1 };
		GPU_RUN_KERNAL(gpu_kl_build_calc_prob, 3, wdims, local_size);
	}

	// find max index
	{
		HIBAG_TIMING(TM_BUILD_OOB_MAXIDX)
		size_t wdims[2] = { gpu_const_local_size, (size_t)build_num_oob };
		size_t local_size[2] = { gpu_const_local_size, 1 };
		GPU_RUN_KERNAL(gpu_kl_build_find_maxprob, 2, wdims, local_size);
	}

	// sync memory
	GPU_MEM_MAP<TGenotype> MG(mem_snpgeno, Num_Sample, true);
	GPU_MEM_MAP<int> MP(mem_build_param, offset_build_param + Num_Sample, true);
	GPU_MEM_MAP<int> MO(mem_build_output, build_num_oob, true);

	TGenotype *pGeno = MG.ptr();
	int *pIdx = MP.ptr() + offset_build_param;
	int *pMaxI = MO.ptr();
	THLAType hla;
	int corrent_cnt = 0;

	for (int i=0; i < build_num_oob; i++)
	{
		size_t k = pMaxI[i] << 1;
		if (k >= 0)
		{
			hla.Allele1 = hla_map_index[k];
			hla.Allele2 = hla_map_index[k+1];
		} else {
			hla.Allele1 = hla.Allele2 = NA_INTEGER;
		}
		corrent_cnt += compare(hla, pGeno[pIdx[i]].aux_hla_type);
	}

	return corrent_cnt;
}


double build_acc_ib()
{
	if (build_num_ib <= 0) return 0;
	if (build_num_ib > mem_sample_nmax)
		throw "Too many sample out of the limit of GPU memory, please contact the package author.";

	cl_int err;
	// initialize
	{
		HIBAG_TIMING(TM_BUILD_IB_CLEAR)
		clear_prob_buffer(msize_prob_buffer * build_num_ib);
		int param[4] = { run_num_haplo, run_num_snp, 0, offset_build_param+Num_Sample };
		GPU_WRITE_MEM(mem_build_param, 0, sizeof(param), param);
	}

	// run OpenCL (calculating probabilities)
	{
		HIBAG_TIMING(TM_BUILD_IB_PROB)
		size_t wdims[3] =
			{ (size_t)wdim_num_haplo, (size_t)wdim_num_haplo, (size_t)build_num_ib };
		size_t local_size[3] =
			{ gpu_local_size_d2, gpu_local_size_d2, 1 };
		GPU_RUN_KERNAL(gpu_kl_build_calc_prob, 3, wdims, local_size);
	}

	// get sum of prob for each sample
	{
		HIBAG_TIMING(TM_BUILD_IB_LOGLIK)
		size_t wdims[2] = { gpu_const_local_size, (size_t)build_num_ib };
		size_t local_size[2] = { gpu_const_local_size, 1 };
		GPU_RUN_KERNAL(gpu_kl_build_sum_prob, 2, wdims, local_size);
	}

	// sum of log likelihood
	double LogLik = 0;
	if (gpu_f64_build_flag)
	{
		GPU_MEM_MAP<double> M(mem_build_output, build_num_ib, true);
		const double *p = M.ptr();
		for (int i=0; i < build_num_ib; i++) LogLik += p[i];
	} else {
		GPU_MEM_MAP<float> M(mem_build_output, build_num_ib, true);
		const float *p = M.ptr();
		for (int i=0; i < build_num_ib; i++) LogLik += p[i];
	}

	// output
	return -2 * LogLik;
}



// ===================================================================== //

/// initialize the internal structure for predicting
void predict_init(int nHLA, int nClassifier, const THaplotype *const pHaplo[],
	const int nHaplo[])
{
	// 64-bit floating-point number or not?
	gpu_f64_pred_flag = Rf_asLogical(get_var_env("flag_pred_f64")) == TRUE;

	// device variables
	cl_int err;
	gpu_context = get_context_env("gpu_context");
	gpu_command_queue = get_command_queue_env("gpu_context");

	// kernels
	gpu_kl_pred_calc    = get_kernel_env("kernel_pred_calc");
	gpu_kl_pred_sumprob = get_kernel_env("kernel_pred_sumprob");
	gpu_kl_pred_addprob = get_kernel_env("kernel_pred_addprob");
	gpu_kl_clear_mem    = get_kernel_env("kernel_clear_mem");

	// assign
	Num_HLA = nHLA;
	Num_Classifier = nClassifier;
	const size_t size_hla = nHLA * (nHLA+1) >> 1;

	// the number of haplotypes among all classifiers in total
	size_t sum_n_haplo=0, max_n_haplo=0;
	for (int i=0; i < nClassifier; i++)
	{
		size_t m = nHaplo[i << 1];
		sum_n_haplo += m;
		if (m > max_n_haplo) max_n_haplo = m;
	}
	wdim_num_haplo = max_n_haplo;
	if (wdim_num_haplo % gpu_local_size_d2)
		wdim_num_haplo = (wdim_num_haplo/gpu_local_size_d2 + 1)*gpu_local_size_d2;

	// GPU memory
	cl_mem mem_rare_freq = get_mem_env(gpu_f64_pred_flag ?
		"mem_exp_log_min_rare_freq64" : "mem_exp_log_min_rare_freq32");

	// memory for SNP genotypes
	GPU_CREATE_MEM(mem_snpgeno, CL_MEM_READ_ONLY,
		sizeof(TGenotype)*nClassifier, NULL);

	// haplotype lists for all classifiers
	vector<int> nhaplo_buf(4*nClassifier);
	const size_t msize_haplo = sizeof(THaplotype)*sum_n_haplo;
	if (gpu_verbose)
		Rprintf("    allocating %lld bytes in GPU ", (long long)msize_haplo);
	GPU_CREATE_MEM(mem_haplo_list, CL_MEM_READ_ONLY, msize_haplo, NULL);
	if (gpu_verbose) Rprintf("[OK]\n");
	{
		GPU_MEM_MAP<THaplotype> M(mem_haplo_list, msize_haplo, false);
		THaplotype *p = M.ptr();
		for (int i=0; i < nClassifier; i++)
		{
			size_t m = nHaplo[i*2];
			nhaplo_buf[i*4 + 0] = m;
			nhaplo_buf[i*4 + 1] = p - M.ptr();
			nhaplo_buf[i*4 + 2] = nHaplo[i*2 + 1];
			memcpy(p, pHaplo[i], sizeof(THaplotype)*m);
			p += m;
		}
	}

	// the numbers of haplotypes
	GPU_CREATE_MEM(mem_pred_haplo_num, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(int)*nhaplo_buf.size(), (void*)&nhaplo_buf[0]);

	// pred_calc_prob -- out_prob
	msize_prob_buffer = size_hla * (gpu_f64_pred_flag ? sizeof(double) : sizeof(float));
	msize_prob_buffer_total = msize_prob_buffer * nClassifier;
	if (gpu_verbose)
		Rprintf("    allocating %lld bytes in GPU ", (long long)msize_prob_buffer_total);
	GPU_CREATE_MEM(mem_prob_buffer, CL_MEM_READ_WRITE, msize_prob_buffer_total, NULL);
	if (gpu_verbose) Rprintf("[OK]\n");

	// pred_calc_addprob -- weight
	GPU_CREATE_MEM(mem_pred_weight, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
		sizeof(double)*nClassifier, NULL);

	// arguments for gpu_kl_pred_calc, pred_calc_prob
	int sz_hla = size_hla;
	GPU_SETARG(gpu_kl_pred_calc, 0, mem_prob_buffer);
	GPU_SETARG(gpu_kl_pred_calc, 1, nHLA);
	GPU_SETARG(gpu_kl_pred_calc, 2, sz_hla);
	GPU_SETARG(gpu_kl_pred_calc, 3, mem_rare_freq);
	GPU_SETARG(gpu_kl_pred_calc, 4, mem_haplo_list);
	GPU_SETARG(gpu_kl_pred_calc, 5, mem_pred_haplo_num);
	GPU_SETARG(gpu_kl_pred_calc, 6, mem_snpgeno);

	// arguments for gpu_kl_pred_sumprob, pred_calc_sumprob
	GPU_SETARG(gpu_kl_pred_sumprob, 0, mem_pred_weight);
	GPU_SETARG(gpu_kl_pred_sumprob, 1, sz_hla);
	GPU_SETARG(gpu_kl_pred_sumprob, 2, mem_prob_buffer);

	// arguments for gpu_kl_pred_addprob, pred_calc_addprob
	GPU_SETARG(gpu_kl_pred_addprob, 0, mem_prob_buffer);
	GPU_SETARG(gpu_kl_pred_addprob, 1, sz_hla);
	GPU_SETARG(gpu_kl_pred_addprob, 2, nClassifier);
	GPU_SETARG(gpu_kl_pred_addprob, 3, mem_pred_weight);
	wdim_pred_addprob = size_hla;
	if (wdim_pred_addprob % gpu_local_size_d1)
		wdim_pred_addprob = (wdim_pred_addprob/gpu_local_size_d1 + 1) * gpu_local_size_d1;

	// arguments for gpu_kl_build_clear_mem
	GPU_SETARG(gpu_kl_clear_mem, 1, mem_prob_buffer);
}


/// finalize the structure for predicting
void predict_done()
{
	GPU_FREE_MEM(mem_haplo_list);
	GPU_FREE_MEM(mem_pred_haplo_num);
	GPU_FREE_MEM(mem_snpgeno);
	GPU_FREE_MEM(mem_prob_buffer);
	GPU_FREE_MEM(mem_pred_weight);
}


/// average the posterior probabilities among classifiers for predicting
void predict_avg_prob(const TGenotype geno[], const double weight[],
	double out_prob[], double out_match[])
{
	const size_t num_size = Num_HLA * (Num_HLA + 1) >> 1;

	// initialize
	cl_int err;
	GPU_WRITE_MEM(mem_snpgeno, 0, sizeof(TGenotype)*Num_Classifier, geno);
	clear_prob_buffer(msize_prob_buffer_total);

	// pred_calc_prob
	{
		size_t wdims[3] =
			{ (size_t)wdim_num_haplo, (size_t)wdim_num_haplo, (size_t)Num_Classifier };
		size_t local_size[3] =
			{ gpu_local_size_d2, gpu_local_size_d2, 1 };
		GPU_RUN_KERNAL(gpu_kl_pred_calc, 3, wdims, local_size);
	}

	// use host to calculate if single-precision
	if (gpu_f64_pred_flag)
	{
		// sum up all probs for each classifier
		{
			// output to mem_pred_weight
			size_t wdims[2] = { (size_t)gpu_const_local_size, (size_t)Num_Classifier };
			size_t local_size[2] = { gpu_const_local_size, 1 };
			GPU_RUN_KERNAL(gpu_kl_pred_sumprob, 2, wdims, local_size);
		}

		// mem_pred_weight
		{
			GPU_MEM_MAP<double> M(mem_pred_weight, sizeof(double)*Num_Classifier, false);
			double psum = 0, *w = M.ptr();
			for (int i=0; i < Num_Classifier; i++)
			{
				psum += w[i];
				w[i] = weight[i] / w[i];
				if (!R_FINITE(w[i])) w[i] = 0;
			}
			if (out_match)
				*out_match = psum / Num_Classifier;
		}

		// sum up all probs among classifiers per HLA genotype
		size_t wdim = wdim_pred_addprob;
		GPU_RUN_KERNAL(gpu_kl_pred_addprob, 1, &wdim, &gpu_local_size_d1);
		GPU_READ_MEM(mem_prob_buffer, sizeof(double)*num_size, out_prob);
		// normalize out_prob
		fmul_f64(out_prob, num_size, 1 / get_sum_f64(out_prob, num_size));

	} else {
		// using double in hosts to improve precision
		GPU_MEM_MAP<float> M(mem_prob_buffer, msize_prob_buffer_total, true);

		memset(out_prob, 0, sizeof(double)*num_size);
		const float *p = M.ptr();
		double psum = 0;

		for (int i=0; i < Num_Classifier; i++, p+=num_size)
		{
			double ss = 0;
			for (size_t k=0; k < num_size; k++)
				ss += p[k];
			if (ss > 0)
			{
				psum += ss;
				double w = weight[i] / ss;
				for (size_t k=0; k < num_size; k++)
					out_prob[k] += p[k] * w;
			}
		}

		*out_match = psum / Num_Classifier;
		fmul_f64(out_prob, num_size, 1 / get_sum_f64(out_prob, num_size));
		return;
	}
}


// ===================================================================== //

/// release the memory buffer object
SEXP gpu_free_memory(SEXP buffer)
{
	if (!Rf_inherits(buffer, "clBuffer") || TYPEOF(buffer) != EXTPTRSXP)
		Rf_error("Not an OpenCL buffer.");
	cl_mem mem = (cl_mem)R_ExternalPtrAddr(buffer);
	if (mem)
	{
		clReleaseMemObject(mem);
		R_ClearExternalPtr(buffer);
		// Rprintf("Release OpenCL memory buffer (%p)\n", (void*)mem);
	}
	return R_NilValue;
}


/// automatically set the work local size for the kernel
SEXP gpu_set_local_size()
{
	cl_device_id dev = get_device_env("gpu_device");
	cl_kernel kernel = get_kernel_env("kernel_clear_mem");

	size_t mem_byte = 0;
	cl_int err = clGetKernelWorkGroupInfo(kernel, dev, CL_KERNEL_WORK_GROUP_SIZE,
		sizeof(mem_byte), &mem_byte, NULL);
	if (err==CL_SUCCESS && mem_byte>64)
	{
		gpu_local_size_d1 = mem_byte;
		if (mem_byte >= 1024)
			gpu_local_size_d2 = 32;
		else if (mem_byte >= 256)
			gpu_local_size_d2 = 16;
		else
			gpu_local_size_d2 = 8;
	} else {
		gpu_local_size_d1 = 64;
		gpu_local_size_d2 = 8;
	}

	if (gpu_verbose)
	{
		Rprintf("    local work size: %d (D1), %dx%d (D2)\n",
			(int)gpu_local_size_d1, (int)gpu_local_size_d2, (int)gpu_local_size_d2);
	}

	return R_NilValue;
}


/// get GPU internal parameters
SEXP gpu_get_param()
{
	cl_int err;
	cl_device_id dev = get_device_env("gpu_device");
	cl_kernel k = get_kernel_env("kernel_clear_mem");

	double gl_mem_sz = R_NaN;
	{
		cl_ulong v;
		err = clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(v), &v, NULL);
		if (err == CL_SUCCESS) gl_mem_sz = v;
	}
	double gl_mem_alloc_sz = R_NaN;
	{
		cl_ulong v;
		err = clGetDeviceInfo(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(v), &v, NULL);
		if (err == CL_SUCCESS) gl_mem_alloc_sz = v;
	}
	int n_unit = NA_INTEGER;
	{
		cl_uint v;
		err = clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(v), &v, NULL);
		if (err == CL_SUCCESS) n_unit = v;
	}

	int ws = get_kernel_param(dev, k, CL_KERNEL_WORK_GROUP_SIZE);
	int mt = get_kernel_param(dev, k, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE);
	SEXP rv_ans = PROTECT(NEW_LIST(5));
	SET_ELEMENT(rv_ans, 0, Rf_ScalarReal(gl_mem_sz));
	SET_ELEMENT(rv_ans, 1, Rf_ScalarReal(gl_mem_alloc_sz));
	SET_ELEMENT(rv_ans, 2, Rf_ScalarInteger(n_unit));
	SET_ELEMENT(rv_ans, 3, Rf_ScalarInteger(ws));
	SET_ELEMENT(rv_ans, 4, Rf_ScalarInteger(mt));
	UNPROTECT(1);
	return rv_ans;
}

/// return EXP_LOG_MIN_RARE_FREQ
SEXP gpu_exp_log_min_rare_freq()
{
	/// the minimum rare frequency to store haplotypes (defined in HIBAG)
	const double MIN_RARE_FREQ = 1e-5;
	const int n = 2 * HIBAG_MAXNUM_SNP_IN_CLASSIFIER + 1;
	SEXP rv_ans = PROTECT(NEW_NUMERIC(n));
	for (int i=0; i < n; i++)
		REAL(rv_ans)[i] = exp(i * log(MIN_RARE_FREQ));
	UNPROTECT(1);
	return rv_ans;
}


/// initialize GPU structure and return a pointer object
SEXP gpu_init_proc(SEXP env)
{
	// check
	if (sizeof(THaplotype) != 32)
	{
		Rf_error("sizeof(THaplotype) should be 32, but receive %d.",
			(int)sizeof(THaplotype));
	}
	if (offsetof(THaplotype, Freq) != 16)
	{
		Rf_error("offsetof(THaplotype, Freq) should be 16, but receive %d.",
			(int)offsetof(THaplotype, Freq));
	}
	if (offsetof(THaplotype, aux.a2.Freq_f32) != 24)
	{
		Rf_error("offsetof(THaplotype, aux.a2.Freq_f32) should be 24, but receive %d.",
			(int)offsetof(THaplotype, aux.a2.Freq_f32));
	}
	if (offsetof(THaplotype, aux.a2.HLA_allele) != 28)
	{
		Rf_error("offsetof(THaplotype, aux.a2.Freq_f32) should be 24, but receive %d.",
			(int)offsetof(THaplotype, aux.a2.HLA_allele));
	}

	if (sizeof(TGenotype) != 48)
	{
		Rf_error("sizeof(TGenotype) should be 48, but receive %d.",
			(int)sizeof(TGenotype));
	}
	if (offsetof(TGenotype, BootstrapCount) != 32)
	{
		Rf_error("offsetof(TGenotype, BootstrapCount) should be 32, but receive %d.",
			(int)offsetof(TGenotype, BootstrapCount));
	}
	if (offsetof(TGenotype, aux_hla_type.Allele1) != 36)
	{
		Rf_error("offsetof(TGenotype, aux_hla_type.Allele1) should be 36, but receive %d.",
			(int)offsetof(TGenotype, aux_hla_type.Allele1));
	}
	if (offsetof(TGenotype, aux_hla_type.Allele2) != 40)
	{
		Rf_error("offsetof(TGenotype, aux_hla_type.Allele2) should be 40, but receive %d.",
			(int)offsetof(TGenotype, aux_hla_type.Allele2));
	}

	// initialize package-wide variables
	packageEnv = env;
	// initialize GPU_Proc
	GPU_Proc.build_init = build_init;
	GPU_Proc.build_done = build_done;
	GPU_Proc.build_set_bootstrap = build_set_bootstrap;
	GPU_Proc.build_set_haplo_geno = build_set_haplo_geno;
	GPU_Proc.build_acc_oob = build_acc_oob;
	GPU_Proc.build_acc_ib = build_acc_ib;
	GPU_Proc.predict_init = predict_init;
	GPU_Proc.predict_done = predict_done;
	GPU_Proc.predict_avg_prob = predict_avg_prob;

	// return the pointer object
	return R_MakeExternalPtr(&GPU_Proc, R_NilValue, R_NilValue);
}

} // extern "C"
