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
	static cl_kernel gpu_kl_build_calc_oob = NULL;
	static cl_kernel gpu_kl_build_calc_ib = NULL;
	static cl_kernel gpu_kl_pred_calc = NULL;
	static cl_kernel gpu_kl_pred_sumprob = NULL;
	static cl_kernel gpu_kl_pred_addprob = NULL;
	static cl_kernel gpu_kl_clear_mem = NULL;

	// OpenCL memory objects

	// parameters, int[] =
	static cl_mem mem_build_idx_oob = NULL;
	static cl_mem mem_build_idx_ib = NULL;

	// SNP genotypes, TGenotype[]
	static cl_mem mem_snpgeno = NULL;

	// haplotype list, THaplotype[]
	static cl_mem mem_haplo_list = NULL;

	// the buffer of probabilities
	// double[nHLA*(nHLA+1)/2][# of samples] -- prob. for classifiers or samples
	static cl_mem mem_prob_buffer = NULL;
	// sizeof(double[nHLA*(nHLA+1)/2]), or sizeof(float[nHLA*(nHLA+1)/2])
	static size_t msize_prob_buffer_each = 0;
	// build:   msize_prob_buffer_total = msize_prob_buffer_each * num_sample
	// predict: msize_prob_buffer_total = msize_prob_buffer_each * num_classifier
	static size_t msize_prob_buffer_total = 0;

	///< an index according to a pair of alleles
	static cl_mem mem_build_hla_idx_map = NULL;

	/// max. index or sum of prob.
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
	static size_t gpu_const_local_size = 64;
	static size_t gpu_local_size_d1 = 64;  // will be determined automatically
	static size_t gpu_local_size_d2 = 8;   // will be determined automatically



	// ===================================================================== //
	// building classifiers

	static int build_num_oob;   ///< the number of out-of-bag samples
	static int build_num_ib;    ///< the number of in-bag samples
	static int run_num_haplo;   ///< the total number of haplotypes
	static int run_num_snp;     ///< the number of SNPs
	static int wdim_num_haplo;  ///< global_work_size for the number of haplotypes


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


	#define GPU_MEM_MAP(VAR, TYPE, m, s, r)    TGPU_Mem_Map<TYPE> VAR(m, s, r, #m)

	/// Map and unmap GPU memory buffer
	template<typename TYPE> struct TGPU_Mem_Map
	{
	public:
		/// constructor
		TGPU_Mem_Map(cl_mem mem, size_t size, bool readonly, const char *fn)
		{
			cl_int err;
			gpu_mem = mem;
			void *p = clEnqueueMapBuffer(gpu_command_queue, gpu_mem, CL_TRUE, // blocking
				CL_MAP_READ | (readonly ? 0 : CL_MAP_WRITE),
				0, size * sizeof(TYPE), 0, NULL, NULL, &err);
			if (!(mem_ptr = (TYPE*)p))
			{
				static char buffer[256];
				if (fn == NULL) fn = "";
				sprintf(buffer,
					"Unable to map '%s' to host memory [%lld bytes] (error: %d, %s)",
					fn, (long long)(size*sizeof(TYPE)), err, err_code(err));
				throw buffer;
			}
		}
		/// destructor
		~TGPU_Mem_Map()
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

extern "C"
{
/// GPU debug information with function name
static const char *gpu_debug_func_name = NULL;

/// OpenCL error message
static const char *gpu_err_msg(const char *txt, int err)
{
	static char buf[1024];
	if (gpu_debug_func_name)
	{
		sprintf(buf, "%s '%s' (error: %d, %s).", txt, gpu_debug_func_name,
			err, err_code(err));
	} else {
		sprintf(buf, "%s (error: %d, %s).", txt, err, err_code(err));
	}
	return buf;
}

/// OpenCL call clFinish
inline static void gpu_finish()
{
	cl_int err = clFinish(gpu_command_queue);
	if (err != CL_SUCCESS)
		throw gpu_err_msg("Failed to call clFinish()", err);
}

/// OpenCL call clReleaseEvent
inline static void gpu_free_events(cl_uint num_events, const cl_event event_list[])
{
	for (cl_uint i=0; i < num_events; i++)
		clReleaseEvent(event_list[i]);
}


// ====  GPU create/free memory buffer  ====

#define GPU_CREATE_MEM(x, flags, size, host_ptr)    \
	gpu_debug_func_name = #x; \
	x = gpu_create_mem(flags, size, host_ptr); \
	gpu_debug_func_name = NULL;

static cl_mem gpu_create_mem(cl_mem_flags flags, size_t size, void *host_ptr)
{
	cl_int err;
	cl_mem mem = clCreateBuffer(gpu_context, flags, size, host_ptr, &err);
	if (!mem)
	{
		static char buf[1024];
		if (gpu_debug_func_name)
		{
			sprintf(buf, "Failed to create memory buffer (%lld bytes) '%s' (error: %d, %s).",
				(long long)size, gpu_debug_func_name, err, err_code(err));
		} else {
			sprintf(buf, "Failed to create memory buffer (%lld bytes) (error: %d, %s).",
				(long long)size, err, err_code(err));
		}
		throw buf;
	}
	return mem;
}

#define GPU_FREE_MEM(x)    \
	{ \
		gpu_debug_func_name = #x; \
		if (x) { \
			cl_int err = clReleaseMemObject(x); \
			if (err != CL_SUCCESS) \
				throw gpu_err_msg("Failed to free memory buffer", err); \
			x = NULL; \
		} \
		gpu_debug_func_name = NULL; \
	}


// ====  GPU memory buffer writing  ====

#define GPU_WRITE_MEM(x, size, ptr)    \
	gpu_debug_func_name = #x; \
	gpu_write_mem(x, true, size, ptr); \
	gpu_debug_func_name = NULL;

#define GPU_WRITE_EVENT(v, x, size, ptr)    \
	gpu_debug_func_name = #x; \
	v = gpu_write_mem(x, false, size, ptr); \
	gpu_debug_func_name = NULL;

static cl_event gpu_write_mem(cl_mem buffer, bool blocking, size_t size,
	const void *ptr)
{
	cl_event event = NULL;
	cl_int err = clEnqueueWriteBuffer(gpu_command_queue, buffer,
		blocking ? CL_TRUE : CL_FALSE, 0, size, ptr, 0, NULL,
		blocking ? NULL : &event);
	if (err != CL_SUCCESS)
		throw gpu_err_msg("Failed to write memory buffer", err);
	return event;
}


#define GPU_READ_MEM(x, offset, size, ptr)    \
	{ \
		cl_int err = clEnqueueReadBuffer(gpu_command_queue, x, CL_TRUE, offset, size, \
			ptr, 0, NULL, NULL); \
		if (err != CL_SUCCESS) \
			throw err_text("Failed to read memory buffer " #x, err); \
	}


// ====  GPU set arguments  ====

#define GPU_SETARG(kernel, i, x)    \
	{ \
		cl_int err = clSetKernelArg(kernel, i, sizeof(x), &x); \
		if (err != CL_SUCCESS) \
			throw err_text("Failed to set kernel (" #kernel ") argument (" #i ")", err); \
	}


// ====  GPU run kernel  ====

#define GPU_RUN_KERNEL(kernel, ndim, wdims, lsize)    \
	{ \
		cl_int err = clEnqueueNDRangeKernel(gpu_command_queue, kernel, ndim, NULL, \
			wdims, lsize, 0, NULL, NULL); \
		if (err != CL_SUCCESS) \
			throw err_text("Failed to run clEnqueueNDRangeKernel() with " #kernel, err); \
		err = clFinish(gpu_command_queue); \
		if (err != CL_SUCCESS) \
			throw err_text("Failed to run clFinish() with " #kernel, err); \
	}

#define GPU_RUN_KERNEL_EVENT(kernel, ndim, wdims, lsize, num_e, ptr_e, out_e)    \
	{ \
		cl_int err = clEnqueueNDRangeKernel(gpu_command_queue, kernel, ndim, NULL, \
			wdims, lsize, num_e, ptr_e, out_e); \
		if (err != CL_SUCCESS) \
			throw err_text("Failed to run clEnqueueNDRangeKernel() with " #kernel, err); \
	}


// ====  GPU clear memory buffer  ====

/// clear the memory buffer 'mem_prob_buffer'
static void clear_prob_buffer(size_t size, cl_event *event)
{
#if defined(CL_VERSION_1_2) && 0
	// don't know why clEnqueueFillBuffer failed without an error return on my AMD Radeon Pro 560X
	// so disable it
	int zero = 0;
	cl_int err = clEnqueueFillBuffer(gpu_command_queue, mem_prob_buffer,
		&zero, sizeof(zero), 0, size, 0, NULL, event);
	if (err != CL_SUCCESS)
	{
		if (event) clReleaseEvent(*event);
		throw err_text("clEnqueueFillBuffer() with mem_prob_buffer failed", err);
	}
	if (!event) gpu_finish();
#else
	if (size >= 4294967296)
		throw "size is too large in clear_prob_buffer().";
	// set parameter
	int n = size / sizeof(int);
	GPU_SETARG(gpu_kl_clear_mem, 0, n);
	size_t wdim = n / gpu_local_size_d1;
	if (n % gpu_local_size_d1) wdim++;
	wdim *= gpu_local_size_d1;
	// run the kernel
	gpu_debug_func_name = "gpu_kl_clear_mem";
	cl_int err = clEnqueueNDRangeKernel(gpu_command_queue, gpu_kl_clear_mem, 1, NULL,
		&wdim, &gpu_local_size_d1, 0, NULL, event);
	if (err != CL_SUCCESS)
		throw gpu_err_msg("Failed to run clEnqueueNDRangeKernel() on", err);
	gpu_debug_func_name = NULL;
	if (!event) gpu_finish();
#endif
}


// =========================================================================

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


// =========================================================================

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
	if (nHLA >= 32768)
		throw "There are too many unique HLA alleles.";

	// initialize
	Num_HLA = nHLA;
	Num_Sample = nSample;
	const int sz_hla = nHLA*(nHLA+1)/2;

	// 64-bit floating-point number or not?
	gpu_f64_build_flag = Rf_asLogical(get_var_env("flag_build_f64")) == TRUE;

	// device variables
	gpu_context = get_context_env("gpu_context");
	gpu_command_queue = get_command_queue_env("gpu_context");

	// kernels
	gpu_kl_build_calc_prob = get_kernel_env("kernel_build_calc_prob");
	gpu_kl_build_calc_oob  = get_kernel_env("kernel_build_calc_oob");
	gpu_kl_build_calc_ib   = get_kernel_env("kernel_build_calc_ib");
	gpu_kl_clear_mem       = get_kernel_env("kernel_clear_mem");

	// GPU memory
	cl_mem mem_rare_freq = get_mem_env(gpu_f64_build_flag ?
		"mem_exp_log_min_rare_freq64" : "mem_exp_log_min_rare_freq32");
	mem_prob_buffer = get_mem_env("mem_prob_buffer");
	msize_prob_buffer_each = sz_hla *
		(gpu_f64_build_flag ? sizeof(double) : sizeof(float));
	mem_build_idx_oob = get_mem_env("mem_build_idx_oob");
	mem_build_idx_ib = get_mem_env("mem_build_idx_ib");
	mem_haplo_list = get_mem_env("mem_haplo_list");
	mem_snpgeno = get_mem_env("mem_snpgeno");
	build_haplo_nmax = Rf_asInteger(get_var_env("build_haplo_nmax"));
	mem_sample_nmax = Rf_asInteger(get_var_env("build_sample_nmax"));
	msize_prob_buffer_total = msize_prob_buffer_each * mem_sample_nmax;
	mem_build_hla_idx_map = get_mem_env("mem_build_hla_idx_map");
	mem_build_output = get_mem_env("mem_build_output");

	{
		// initialize mem_build_hla_idx_map
		GPU_MEM_MAP(M, int, mem_build_hla_idx_map, sz_hla, false);
		int *p = M.ptr();
		for (int h1=0; h1 < nHLA; h1++)
		{
			for (int h2=h1; h2 < nHLA; h2++)
				*p++ = h1 | (h2 << 16);
		}
	}

	// arguments for build_calc_prob
	int zero = 0;
	GPU_SETARG(gpu_kl_build_calc_prob, 0, mem_prob_buffer);
	GPU_SETARG(gpu_kl_build_calc_prob, 1, mem_rare_freq);
	GPU_SETARG(gpu_kl_build_calc_prob, 2, nHLA);
	GPU_SETARG(gpu_kl_build_calc_prob, 3, sz_hla);
	GPU_SETARG(gpu_kl_build_calc_prob, 4, zero);  // n_haplo
	GPU_SETARG(gpu_kl_build_calc_prob, 5, zero);  // n_snp
	GPU_SETARG(gpu_kl_build_calc_prob, 6, zero);  // start_sample_idx
	GPU_SETARG(gpu_kl_build_calc_prob, 7, mem_build_idx_oob);  // mem_build_idx_oob or mem_build_idx_ib
	GPU_SETARG(gpu_kl_build_calc_prob, 8, mem_haplo_list);
	GPU_SETARG(gpu_kl_build_calc_prob, 9, mem_snpgeno);

	// arguments for gpu_kl_build_calc_oob (out-of-bag)
	GPU_SETARG(gpu_kl_build_calc_oob, 0, mem_build_output);
	GPU_SETARG(gpu_kl_build_calc_oob, 1, zero);  // start_sample_idx
	GPU_SETARG(gpu_kl_build_calc_oob, 2, sz_hla);
	GPU_SETARG(gpu_kl_build_calc_oob, 3, mem_prob_buffer);
	GPU_SETARG(gpu_kl_build_calc_oob, 4, mem_build_hla_idx_map);
	GPU_SETARG(gpu_kl_build_calc_oob, 5, mem_build_idx_oob);
	GPU_SETARG(gpu_kl_build_calc_oob, 6, mem_snpgeno);

	// arguments for gpu_kl_build_calc_ib (in-bag)
	GPU_SETARG(gpu_kl_build_calc_ib, 0, mem_build_output);
	GPU_SETARG(gpu_kl_build_calc_ib, 1, zero);  // start_sample_idx
	GPU_SETARG(gpu_kl_build_calc_ib, 2, nHLA);
	GPU_SETARG(gpu_kl_build_calc_ib, 3, sz_hla);
	GPU_SETARG(gpu_kl_build_calc_ib, 4, mem_prob_buffer);
	GPU_SETARG(gpu_kl_build_calc_ib, 5, mem_build_idx_ib);
	GPU_SETARG(gpu_kl_build_calc_ib, 6, mem_snpgeno);

	// arguments for gpu_kl_build_clear_mem
	GPU_SETARG(gpu_kl_clear_mem, 1, mem_prob_buffer);
}


void build_done()
{
	gpu_kl_build_calc_prob = gpu_kl_build_calc_oob =
		gpu_kl_build_calc_ib = gpu_kl_clear_mem = NULL;
	mem_build_idx_oob = mem_build_idx_ib = mem_snpgeno = mem_build_output =
		mem_haplo_list = mem_build_hla_idx_map = mem_prob_buffer = NULL;
}

void build_set_bootstrap(const int oob_cnt[])
{
	GPU_MEM_MAP(Moob, int, mem_build_idx_oob, Num_Sample, false);
	GPU_MEM_MAP(Mib, int, mem_build_idx_ib, Num_Sample, false);
	int *p_oob = Moob.ptr();
	int *p_ib  = Mib.ptr();
	build_num_oob = build_num_ib = 0;
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

	run_num_snp = n_snp;
	run_num_haplo = wdim_num_haplo = n_haplo;
	if (wdim_num_haplo % gpu_local_size_d2)
		wdim_num_haplo = (wdim_num_haplo/gpu_local_size_d2 + 1)*gpu_local_size_d2;

	const size_t sz_haplo = sizeof(THaplotype)*n_haplo;
	GPU_WRITE_MEM(mem_haplo_list, sz_haplo, (void*)haplo);

	const size_t sz_geno = sizeof(TGenotype)*Num_Sample;
	GPU_WRITE_MEM(mem_snpgeno, sz_geno, (void*)geno);
}

int build_acc_oob()
{
	if (build_num_oob <= 0) return 0;
	if (build_num_oob > mem_sample_nmax)
		throw "Too many sample out of the limit of GPU memory, please contact the package author.";

	// initialize
	cl_event events[4];
	clear_prob_buffer(msize_prob_buffer_each * build_num_oob, &events[0]);

	// calculate probabilities
	{
		int zero = 0;
		GPU_SETARG(gpu_kl_build_calc_prob, 4, run_num_haplo);  // n_haplo
		GPU_SETARG(gpu_kl_build_calc_prob, 5, run_num_snp);    // n_snp
		GPU_SETARG(gpu_kl_build_calc_prob, 6, zero);           // start_sample_idx
		GPU_SETARG(gpu_kl_build_calc_prob, 7, mem_build_idx_oob);
		size_t wdims[3] =
			{ (size_t)build_num_oob, (size_t)wdim_num_haplo, (size_t)wdim_num_haplo };
		size_t local_size[3] =
			{ 1, gpu_local_size_d2, gpu_local_size_d2 };
		GPU_RUN_KERNEL_EVENT(gpu_kl_build_calc_prob, 3, wdims, local_size,
			1, events, &events[1]);
	}

	// calculate OOB error count
	{
		int zero = 0;  // initialize total error count
		GPU_WRITE_EVENT(events[2], mem_build_output, sizeof(zero), &zero);

		GPU_SETARG(gpu_kl_build_calc_oob, 1, zero);  // start_sample_idx
		size_t wdims[2] = { gpu_const_local_size, (size_t)build_num_oob };
		size_t local_size[2] = { gpu_const_local_size, 1 };
		GPU_RUN_KERNEL_EVENT(gpu_kl_build_calc_oob, 2, wdims, local_size,
			2, &events[1], &events[3]);
	}

	// host waits for GPU
	gpu_finish();
	gpu_free_events(4, events);

	// read output
	int err_cnt;
	GPU_READ_MEM(mem_build_output, 0, sizeof(int), &err_cnt);
	return build_num_oob*2 - err_cnt;
}

double build_acc_ib()
{
	if (build_num_ib <= 0) return 0;
	if (build_num_ib > mem_sample_nmax)
		throw "Too many sample out of the limit of GPU memory, please contact the package author.";

	// initialize
	cl_event events[3];
	clear_prob_buffer(msize_prob_buffer_each * build_num_ib, &events[0]);

	// run OpenCL (calculating probabilities)
	{
		int zero = 0;
		// n_haplo (ARG4) & n_snp (ARG5) are set in build_acc_oob()
		GPU_SETARG(gpu_kl_build_calc_prob, 6, zero);  // start_sample_idx
		GPU_SETARG(gpu_kl_build_calc_prob, 7, mem_build_idx_ib);
		size_t wdims[3] =
			{ (size_t)build_num_ib, (size_t)wdim_num_haplo, (size_t)wdim_num_haplo };
		size_t local_size[3] =
			{ 1, gpu_local_size_d2, gpu_local_size_d2 };
		GPU_RUN_KERNEL_EVENT(gpu_kl_build_calc_prob, 3, wdims, local_size,
			1, events, &events[1]);
	}

	// get sum of prob for each sample
	{
		size_t wdims[2] = { gpu_const_local_size, (size_t)build_num_ib };
		size_t local_size[2] = { gpu_const_local_size, 1 };
		GPU_RUN_KERNEL_EVENT(gpu_kl_build_calc_ib, 2, wdims, local_size,
			1, &events[1], &events[2]);
	}

	// host waits for GPU
	gpu_finish();
	gpu_free_events(3, events);

	// sum of log likelihood
	double LogLik = 0;
	if (gpu_f64_build_flag)
	{
		GPU_MEM_MAP(M, double, mem_build_output, build_num_ib, true);
		const double *p = M.ptr();
		for (int i=0; i < build_num_ib; i++) LogLik += p[i];
	} else {
		GPU_MEM_MAP(M, float, mem_build_output, build_num_ib, true);
		const float *p = M.ptr();
		for (int i=0; i < build_num_ib; i++) LogLik += p[i];
	}

	// output
	return -2 * LogLik;
}



// ===================================================================== //

/// initialize the internal structure for predicting
void predict_init(int nHLA, int nClassifier, const THaplotype *const pHaplo[],
	const int nHaplo[], const int nSNP[])
{
	// 64-bit floating-point number or not?
	gpu_f64_pred_flag = Rf_asLogical(get_var_env("flag_pred_f64")) == TRUE;

	// device variables
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
		size_t m = nHaplo[i];
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
	GPU_CREATE_MEM(mem_snpgeno, CL_MEM_READ_ONLY, sizeof(TGenotype)*nClassifier, NULL);

	// haplotype lists for all classifiers
	vector<int> nhaplo_buf(4*nClassifier);
	const size_t msize_haplo = sizeof(THaplotype)*sum_n_haplo;
	if (gpu_verbose)
		Rprintf("    allocating %lld bytes in GPU ", (long long)msize_haplo);
	GPU_CREATE_MEM(mem_haplo_list, CL_MEM_READ_ONLY, msize_haplo, NULL);
	if (gpu_verbose) Rprintf("[OK]\n");
	{
		GPU_MEM_MAP(M, THaplotype, mem_haplo_list, sum_n_haplo, false);
		THaplotype *p = M.ptr();
		for (int i=0; i < nClassifier; i++)
		{
			size_t m = nHaplo[i];
			nhaplo_buf[i*4 + 0] = m;            // # of haplotypes
			nhaplo_buf[i*4 + 1] = p - M.ptr();  // starting index
			nhaplo_buf[i*4 + 2] = nSNP[i];      // # of SNPs
			memcpy(p, pHaplo[i], sizeof(THaplotype)*m);
			p += m;
		}
	}

	// the numbers of haplotypes
	GPU_CREATE_MEM(mem_pred_haplo_num, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(int)*nhaplo_buf.size(), (void*)&nhaplo_buf[0]);

	// pred_calc_prob -- out_prob
	msize_prob_buffer_each = size_hla *
		(gpu_f64_pred_flag ? sizeof(double) : sizeof(float));
	msize_prob_buffer_total = msize_prob_buffer_each * nClassifier;
	if (gpu_verbose)
		Rprintf("    allocating %lld bytes in GPU ", (long long)msize_prob_buffer_total);
	GPU_CREATE_MEM(mem_prob_buffer, CL_MEM_READ_WRITE, msize_prob_buffer_total, NULL);
	if (gpu_verbose) Rprintf("[OK]\n");

	// pred_calc_addprob -- weight
	GPU_CREATE_MEM(mem_pred_weight, CL_MEM_READ_WRITE,
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
	cl_event events[4];

	// initialize
	clear_prob_buffer(msize_prob_buffer_total, &events[0]);
	GPU_WRITE_EVENT(events[1], mem_snpgeno, sizeof(TGenotype)*Num_Classifier, geno);

	// pred_calc_prob
	{
		size_t wdims[3] =
			{ (size_t)Num_Classifier, (size_t)wdim_num_haplo, (size_t)wdim_num_haplo };
		size_t local_size[3] =
			{ 1, gpu_local_size_d2, gpu_local_size_d2 };
		GPU_RUN_KERNEL_EVENT(gpu_kl_pred_calc, 3, wdims, local_size,
			2, events, &events[2]);
	}

	// use host to calculate if single-precision
	if (gpu_f64_pred_flag)
	{
		// sum up all probs for each classifier
		{
			// output to mem_pred_weight
			size_t wdims[2] = { (size_t)gpu_const_local_size, (size_t)Num_Classifier };
			size_t local_size[2] = { gpu_const_local_size, 1 };
			GPU_RUN_KERNEL_EVENT(gpu_kl_pred_sumprob, 2, wdims, local_size,
				1, &events[2], &events[3]);
		}

		// host waits for GPU
		gpu_finish();
		gpu_free_events(4, events);

		// mem_pred_weight
		{
			GPU_MEM_MAP(M, double, mem_pred_weight, Num_Classifier, false);
			double *w = M.ptr();
			double sum_matching=0, num_matching=0;
			for (int i=0; i < Num_Classifier; i++)
			{
				if (weight[i] <= 0) continue;
				sum_matching += w[i] * weight[i];
				num_matching += weight[i];
				if (w[i] > 0)
					w[i] = weight[i] / w[i];
			}
			if (out_match)
				*out_match = sum_matching / num_matching;
		}

		// sum up all probs among classifiers per HLA genotype
		{
			size_t wdim = wdim_pred_addprob;
			GPU_RUN_KERNEL(gpu_kl_pred_addprob, 1, &wdim, &gpu_local_size_d1);
		}

		GPU_READ_MEM(mem_prob_buffer, 0, sizeof(double)*num_size, out_prob);
		// normalize out_prob
		fmul_f64(out_prob, num_size, 1 / get_sum_f64(out_prob, num_size));

	} else {
		// host waits for GPU
		gpu_finish();
		gpu_free_events(3, events);

		// using double in hosts to improve precision
		GPU_MEM_MAP(M, float, mem_prob_buffer,
			msize_prob_buffer_total/sizeof(float), true);

		memset(out_prob, 0, sizeof(double)*num_size);
		const float *p = M.ptr();
		double sum_matching=0, num_matching=0;
		for (int i=0; i < Num_Classifier; i++, p+=num_size)
		{
			if (weight[i] <= 0) continue;
			double ss = 0;
			for (size_t k=0; k < num_size; k++)
				ss += p[k];
			sum_matching += ss * weight[i];
			num_matching += weight[i];
			if (ss > 0)
			{
				double w = weight[i] / ss;
				for (size_t k=0; k < num_size; k++)
					out_prob[k] += p[k] * w;
			}
		}
		if (out_match)
			*out_match = sum_matching / num_matching;
		fmul_f64(out_prob, num_size, 1 / get_sum_f64(out_prob, num_size));
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

	cl_int err;
	size_t max_wz[128];
	err = clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_wz),
		max_wz, NULL);
	if (err != CL_SUCCESS) max_wz[0] = max_wz[1] = max_wz[2] = 65536;

	size_t mem_byte = 0;
	err = clGetKernelWorkGroupInfo(kernel, dev, CL_KERNEL_WORK_GROUP_SIZE,
		sizeof(mem_byte), &mem_byte, NULL);
	if (err==CL_SUCCESS && mem_byte>64)
	{
		gpu_local_size_d1 = mem_byte;
		if (mem_byte >= 4096)
			gpu_local_size_d2 = 64;
		else if (mem_byte >= 1024)
			gpu_local_size_d2 = 32;
		else if (mem_byte >= 256)
			gpu_local_size_d2 = 16;
		else
			gpu_local_size_d2 = 8;
	} else {
		gpu_local_size_d1 = 64;
		gpu_local_size_d2 = 8;
	}

	if (gpu_local_size_d1 > max_wz[0])
		gpu_local_size_d1 = max_wz[0];
	if (gpu_local_size_d2 > max_wz[1])
		gpu_local_size_d2 = max_wz[1];

	gpu_const_local_size = 64;
	if (gpu_const_local_size > gpu_local_size_d1)
		gpu_const_local_size = gpu_local_size_d1;

	if (gpu_local_size_d2 == 1) // it is a CPU (very likely)
	{
		gpu_local_size_d1 = 1;
		gpu_const_local_size = 1;
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

	#define GET_DEV_INFO(TYPE, PARAM, VAR)    { \
		TYPE v; \
		err = clGetDeviceInfo(dev, PARAM, sizeof(v), &v, NULL); \
		if (err == CL_SUCCESS) VAR = v; \
	}

	double gl_mem_sz = R_NaN;
	GET_DEV_INFO(cl_ulong, CL_DEVICE_GLOBAL_MEM_SIZE, gl_mem_sz)
	double gl_mem_alloc_sz = R_NaN;
	GET_DEV_INFO(cl_ulong, CL_DEVICE_MAX_MEM_ALLOC_SIZE, gl_mem_alloc_sz)
	int gl_n_unit = NA_INTEGER;
	GET_DEV_INFO(cl_uint, CL_DEVICE_MAX_COMPUTE_UNITS, gl_n_unit)
	int gl_max_worksize = NA_INTEGER;
	GET_DEV_INFO(size_t, CL_DEVICE_MAX_WORK_GROUP_SIZE, gl_max_worksize)
	int gl_max_workdim = NA_INTEGER;
	GET_DEV_INFO(cl_uint, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, gl_max_workdim)

	size_t gl_max_work_item_sizes[128] =
		{ (size_t)NA_INTEGER, (size_t)NA_INTEGER, (size_t)NA_INTEGER };
	clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(gl_max_work_item_sizes),
		gl_max_work_item_sizes, NULL);

	int ws = get_kernel_param(dev, k, CL_KERNEL_WORK_GROUP_SIZE);
	int mt = get_kernel_param(dev, k, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE);

	SEXP rv_ans = PROTECT(NEW_LIST(8));
	SET_ELEMENT(rv_ans, 0, Rf_ScalarReal(gl_mem_sz));
	SET_ELEMENT(rv_ans, 1, Rf_ScalarReal(gl_mem_alloc_sz));
	SET_ELEMENT(rv_ans, 2, Rf_ScalarInteger(gl_n_unit));
	SET_ELEMENT(rv_ans, 3, Rf_ScalarInteger(gl_max_worksize));
	SET_ELEMENT(rv_ans, 4, Rf_ScalarInteger(gl_max_workdim));

	SEXP p_wis = NEW_INTEGER(3);
	INTEGER(p_wis)[0] = gl_max_work_item_sizes[0];
	INTEGER(p_wis)[1] = gl_max_work_item_sizes[1];
	INTEGER(p_wis)[2] = gl_max_work_item_sizes[2];
	SET_ELEMENT(rv_ans, 5, p_wis);

	SET_ELEMENT(rv_ans, 6, Rf_ScalarInteger(ws));
	SET_ELEMENT(rv_ans, 7, Rf_ScalarInteger(mt));
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
