// ===============================================================
//
// HIBAG.gpu R package (GPU-based implementation for the HIBAG package)
// Copyright (C) 2017	Xiuwen Zheng (zhengx@u.washington.edu)
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


#include <stdint.h>
#include <string.h>
#include <cstdlib>
#include <cmath>
#include <vector>

#ifdef __APPLE__
#	include <OpenCL/opencl.h>
#else
#	include <CL/opencl.h>
#endif

#define USE_RINTERNALS 1
#include <Rinternals.h>
#include <Rdefines.h>


// Streaming SIMD Extensions, SSE, SSE2, SSE4_2 (POPCNT)

#ifdef __SSE__
#	include <xmmintrin.h>  // SSE
#endif

#ifdef __SSE2__
#	include <emmintrin.h>  // SSE2
#endif


// disable timing
// #define HIBAG_ENABLE_TIMING

#ifdef HIBAG_ENABLE_TIMING
#   include <time.h>
#endif



#define GPU_CREATE_MEM(x, flag, size, ptr)	  \
	x = clCreateBuffer(gpu_context, flag, size, ptr, &err); \
	if (!x) throw err_text("Unable to create buffer " #x, err);

#define GPU_MAP_MEM(x, ptr, size)	 \
	ptr = clEnqueueMapBuffer(gpu_commands, x, CL_TRUE, \
		CL_MAP_READ | CL_MAP_WRITE, 0, size, 0, NULL, NULL, &err); \
	if (!ptr) throw err_text("Unable to map buffer to host memory " #x, err);

#define GPU_UNMAP_MEM(x, ptr)	 \
	err = clEnqueueUnmapMemObject(gpu_commands, x, ptr, 0, NULL, NULL); \
	if (err != CL_SUCCESS) \
		throw err_text("Failed to unmap memory buffer " #x, err); \
	ptr = NULL;

#define GPU_READ_MEM(x, size, ptr)	  \
	err = clEnqueueReadBuffer(gpu_commands, x, CL_TRUE, 0, size, ptr, 0, NULL, NULL); \
	if (err != CL_SUCCESS) \
		throw err_text("Failed to read memory buffer " #x, err);

#define GPU_WRITE_MEM(x, offset, size, ptr)	   \
	err = clEnqueueWriteBuffer(gpu_commands, x, CL_TRUE, offset, size, ptr, 0, NULL, NULL); \
	if (err != CL_SUCCESS) \
		throw err_text("Failed to write memory buffer " #x, err);

#define GPU_SETARG(kernel, i, x)	\
	err = clSetKernelArg(kernel, i, sizeof(x), &x); \
	if (err != CL_SUCCESS) \
		throw err_text("Failed to set kernel (" #kernel ") argument (" #i ")", err);

#define GPU_RUN_KERNAL(kernel, ndim, wdims, lsize)	  \
	err = clEnqueueNDRangeKernel(gpu_commands, kernel, ndim, NULL, wdims, lsize, 0, NULL, NULL); \
	if (err != CL_SUCCESS) \
		throw err_text("Failed to run clEnqueueNDRangeKernel() with " #kernel, err); \
	err = clFinish(gpu_commands); \
	if (err != CL_SUCCESS) \
		throw err_text("Failed to run clFinish() with " #kernel, err);

#define GPU_FREE_MEM(x)	   if (x) { \
		cl_int err = clReleaseMemObject(x); \
		if (err != CL_SUCCESS) \
			throw err_text("Failed to free memory buffer " #x, err); \
		x = NULL; \
	}

#define GPU_FREE_COM(x)	   if (x) { \
		cl_int err = clReleaseCommandQueue(x); \
		if (err != CL_SUCCESS) \
			throw err_text("Failed to release command queue " #x, err); \
		x = NULL; \
	}


#if defined(CL_VERSION_1_2)
	#define GPU_ZERO_FILL(x, size)    { \
		size_t zero = 1; \
		err = clEnqueueFillBuffer(gpu_commands, x, &zero, 1, 0, size, 0, NULL, NULL); \
		if (err != CL_SUCCESS) \
			throw err_text("clEnqueueFillBuffer() with " #x " failed", err); \
	}
#else
	#define GPU_ZERO_FILL(x, size)    { \
		void *ptr; \
		GPU_MAP_MEM(x, ptr, size); \
		memset(ptr, 0, size); \
		GPU_UNMAP_MEM(x, ptr); \
	}
#endif



namespace HLA_LIB
{
	using namespace std;

	/// Define unsigned integers
	typedef uint8_t		UINT8;

	/// The max number of SNP markers in an individual classifier.
	//	Don't modify this value since the code is optimized for this value!!!
	const size_t HIBAG_MAXNUM_SNP_IN_CLASSIFIER = 128;

	/// The max number of UTYPE for packed SNP genotypes.
	const size_t HIBAG_PACKED_UTYPE_MAXNUM =
		HIBAG_MAXNUM_SNP_IN_CLASSIFIER / (8*sizeof(UINT8));


	// ===================================================================== //
	// ========						Description						========
	//
	// Packed SNP storage strategy is used for faster matching
	//
	// HLA allele: start from 0
	//
	// THaplotype: packed SNP alleles (little endianness):
	//	   (s8 s7 s6 s5 s4 s3 s2 s1)
	//	   the 1st allele: (s1), the 2nd allele: (s2), ...
	//	   SNP allele: 0 (B allele), 1 (A allele)
	//
	// TGenotype: packed SNP genotype (little endianness):
	//	   array_1 = (s1_8 s1_7 s1_6 s1_5 s1_4 s1_3 s1_2 s1_1),
	//	   array_2 = (s2_8 s2_7 s2_6 s2_5 s2_4 s2_3 s2_2 s2_1),
	//	   array_3 = (s3_8 s3_7 s3_6 s3_5 s3_4 s3_3 s3_2 s3_1)
	//	   the 1st genotype: (s1_1 s2_1 s3_1),
	//	   the 2nd genotype: (s1_1 s2_1 s3_1), ...
	//	   SNP genotype: 0 (BB) -- (s1_1=0 s2_1=0 s3_1=1),
	//					 1 (AB) -- (s1_1=1 s2_1=0 s3_1=1),
	//					 2 (AA) -- (s1_1=1 s2_1=1 s3_1=1),
	//					 -1 or other value (missing)
	//							-- (s1_1=0 s2_1=0 s3_1=0)
	//
	// ========														========
	// ===================================================================== //

	/// Packed SNP haplotype structure: 8 alleles in a byte
	struct THaplotype
	{
		/// packed SNP alleles
		UINT8 PackedHaplo[HIBAG_PACKED_UTYPE_MAXNUM];
		/// haplotype frequency
		double Freq;
		/// auxiliary variables, sizeof(THaplotype)=32
		union type_aux
		{
			double OldFreq;	 /// old haplotype frequency
			struct type_aux2 {
				float Freq_f32;	 /// 32-bit haplotype frequency
				int HLA_allele;	 /// the associated HLA allele
			} a2;
		} aux;
	};


	/// A pair of HLA alleles
	struct THLAType
	{
		int Allele1;  //< the first HLA allele
		int Allele2;  //< the second HLA allele
	};


	/// Packed bi-allelic SNP genotype structure: 8 SNPs in a byte
	struct TGenotype
	{
		/// packed SNP genotypes, allele 1
		UINT8 PackedSNP1[HIBAG_PACKED_UTYPE_MAXNUM];
		/// packed SNP genotypes, allele 2
		UINT8 PackedSNP2[HIBAG_PACKED_UTYPE_MAXNUM];
		/// packed SNP genotypes, missing flag
		UINT8 PackedMissing[HIBAG_PACKED_UTYPE_MAXNUM];

		/// the count in the bootstrapped data
		int BootstrapCount;

		/// auxiliary correct HLA type
		THLAType aux_hla_type;
		/// auxiliary integer to make sizeof(TGenotype)=64
		int aux_temp;
	};


	/// Pointer to the structure of functions using GPU
	struct TypeGPUExtProc
	{
		/// initialize the internal structure for building a model
		void (*build_init)(int nHLA, int nSample);
		/// finalize the structure for building a model
		void (*build_done)();
		/// initialize bottstrapping
		void (*build_set_bootstrap)(const int oob_cnt[]);
		/// initialize haplotypes and SNPs genotypes
		void (*build_set_haplo_geno)(const THaplotype haplo[], int n_haplo,
			const TGenotype geno[], int n_snp);
		/// calculate the out-of-bag accuracy (the number of correct alleles)
		int (*build_acc_oob)();
		/// calculate the in-bag log likelihood
		double (*build_acc_ib)();

		/// initialize the internal structure for predicting
		//    nHaplo[2*nClassifier]:
		//    nHaplo[0] -- total # of haplotypes
		//    nHaplo[1] -- # of SNPs
		void (*predict_init)(int nHLA, int nClassifier,
			const THaplotype *const pHaplo[], const int nHaplo[]);
		/// finalize the structure for predicting
		void (*predict_done)();
		/// average the posterior probabilities among classifiers for predicting
		void (*predict_avg_prob)(const TGenotype geno[], const double weight[],
			double out_prob[], double out_match[]);
	} GPU_Proc;


	// used for work-group size (1-dim and 2-dim)
	const size_t gpu_local_size_d1 = 64;
	const size_t gpu_local_size_d2 = 8;

	// the number of unique HLA alleles
	static int Num_HLA;
	// the number of individual classifiers
	static int Num_Classifier;
	// the number of samples
	static int Num_Sample;

	// OpenCL device variables
	static cl_context gpu_context = NULL;
	static cl_command_queue gpu_commands = NULL;

	// OpenCL kernel functions
	static cl_kernel gpu_kernel	 = NULL;
	static cl_kernel gpu_kernel2 = NULL;
	static cl_kernel gpu_kernel3 = NULL;
	static cl_kernel gpu_kernel_clear = NULL;

	// flags for usage of double or single precision
	static bool gpu_f64_build_flag = false;
	static bool gpu_f64_pred_flag = false;

	// haplotypes of all classifiers
	static cl_mem mem_exp_log_min_rare_freq = NULL;
	// haplotype list
	static cl_mem mem_haplo_list = NULL;
	// SNP genotypes
	static cl_mem mem_snpgeno = NULL;

	// the buffer of numeric values
	// double[nHLA*(nHLA+1)/2][] -- prob. for each classifier
	static cl_mem mem_prob_buffer = NULL;
	static size_t msize_probbuf_each = 0;
	static size_t msize_probbuf_total = 0;


	// ===================================================================== //
	// building classifiers

	static SEXP kernel_build_prob = NULL;
	static SEXP kernel_build_find_maxprob = NULL;
	static SEXP kernel_build_sum_prob = NULL;
	static SEXP kernel_build_clear = NULL;

	// parameters
	static cl_mem mem_build_param = NULL;
	// parameter offset
	static const int offset_build_param = 5;

	// max. index or sum of prob.
	static cl_mem mem_build_output = NULL;

	// the max number of samples can be hold in mem_prob_buffer
	static int mem_nmax_sample = 0;
	// the max number of haplotypes can be hold in mem_haplo_list
	static int mem_build_haplo_nmax = 0;

	static int num_oob, num_ib, num_haplo, num_snp;
	static vector<int> hla_map_index;


	// ===================================================================== //
	// prediction for calculating the posterior probabilities

	static SEXP kernel_predict = NULL;
	static SEXP kernel_predict_sumprob = NULL;
	static SEXP kernel_predict_addprob = NULL;

	// num of haplotypes and SNPs for each classifier
	static cl_mem mem_pred_haplo_num = NULL;

	// classifier weight
	static cl_mem mem_pred_weight = NULL;

	static int wdim_n_haplo = 0;
	static int wdim_pred_addprob = 0;
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

/// Frequency Calculation
#define FREQ_MUTANT(p, cnt)	   ((p) * EXP_LOG_MIN_RARE_FREQ[cnt]);

/// the minimum rare frequency to store haplotypes
static const double MIN_RARE_FREQ = 1e-5;

/// exp(cnt * log(MIN_RARE_FREQ)), cnt is the hamming distance
static double EXP_LOG_MIN_RARE_FREQ[HIBAG_MAXNUM_SNP_IN_CLASSIFIER*2];
static float  EXP_LOG_MIN_RARE_FREQ_f32[HIBAG_MAXNUM_SNP_IN_CLASSIFIER*2];



// get the OpenCL device ID
static cl_device_id getDeviceID(SEXP device)
{
	if (!Rf_inherits(device, "clDeviceID") || TYPEOF(device) != EXTPTRSXP)
		Rf_error("invalid device");
	return ((cl_device_id*)R_ExternalPtrAddr(device))[0];
}

// get the OpenCL kernel pointer from an R object
static cl_kernel get_kernel(SEXP k)
{
	if (!Rf_inherits(k, "clKernel") || TYPEOF(k) != EXTPTRSXP)
		throw "Invalid OpenCL kernel.";
	return (cl_kernel)R_ExternalPtrAddr(k);
}


static void clFreeContext(SEXP ctx)
{
	clReleaseContext((cl_context)R_ExternalPtrAddr(ctx));
}

static SEXP mkContext(cl_context ctx)
{
	SEXP ptr;
	ptr = PROTECT(R_MakeExternalPtr(ctx, R_NilValue, R_NilValue));
	R_RegisterCFinalizerEx(ptr, clFreeContext, TRUE);
	Rf_setAttrib(ptr, R_ClassSymbol, mkString("clContext"));
	UNPROTECT(1);
	return ptr;
}

static void clFreeKernel(SEXP k)
{
	clReleaseKernel((cl_kernel)R_ExternalPtrAddr(k));
}

static SEXP mkKernel(cl_kernel k)
{
	SEXP ptr;
	ptr = PROTECT(R_MakeExternalPtr(k, R_NilValue, R_NilValue));
	R_RegisterCFinalizerEx(ptr, clFreeKernel, TRUE);
	Rf_setAttrib(ptr, R_ClassSymbol, mkString("clKernel"));
	UNPROTECT(1);
	return ptr;
}


SEXP gpu_build_kernel(SEXP device, SEXP k_name, SEXP code, SEXP prec)
{
	cl_context ctx;
	int err;
	SEXP sctx;
	cl_device_id device_id = getDeviceID(device);
	cl_program program;

	if (TYPEOF(k_name) != STRSXP || LENGTH(k_name) < 1)
		Rf_error("invalid kernel name(s)");
	if (TYPEOF(code) != STRSXP || LENGTH(code) < 1)
		Rf_error("invalid kernel code");
	if (TYPEOF(prec) != STRSXP || LENGTH(prec) != 1)
		Rf_error("invalid precision specification");
	ctx = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (!ctx)
		Rf_error("clCreateContext failed.");
	sctx = PROTECT(mkContext(ctx));
	{
		int sn = LENGTH(code), i;
		const char **cptr;
		cptr = (const char **) malloc(sizeof(char*) * sn);
		for (i = 0; i < sn; i++)
			cptr[i] = CHAR(STRING_ELT(code, i));
		program = clCreateProgramWithSource(ctx, sn, cptr, NULL, &err);
		free(cptr);
		if (!program)
			Rf_error("clCreateProgramWithSource failed");
	}
	
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		size_t len;
		char buffer[2048];
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
			sizeof(buffer), buffer, &len);
		clReleaseProgram(program);
		Rf_error("clGetProgramBuildInfo failed: %s", buffer);
	}

	cl_kernel kernel[length(k_name)];
	for (int i=0; i < length(k_name); i++)
	{
		kernel[i] = clCreateKernel(program, CHAR(STRING_ELT(k_name, i)), &err);
		if (!kernel[i])
			Rf_error("clCreateKernel failed.");
	}

	clReleaseProgram(program);

	SEXP rv_ans = PROTECT(NEW_LIST(length(k_name)));
	for (int i=0; i < length(k_name); i++)
	{
		SEXP sk = PROTECT(mkKernel(kernel[i]));
		Rf_setAttrib(sk, Rf_install("device"), device);
		Rf_setAttrib(sk, Rf_install("precision"), prec);
		Rf_setAttrib(sk, Rf_install("context"), sctx);
		Rf_setAttrib(sk, Rf_install("name"), mkString(CHAR(STRING_ELT(k_name, i))));
		SET_ELEMENT(rv_ans, i, sk);
		UNPROTECT(1);
	}
	UNPROTECT(2);
	return rv_ans;
}




// OpenCL error message
static const char *err_text(const char *txt, int err)
{
	static char buf[1024];
	sprintf(buf, "%s (error: %d).", txt, err);
	return buf;
}


// get the sum of double array
static double get_sum_f64(const double *p, size_t n)
{
#ifdef __SSE2__
	__m128d a, s = _mm_setzero_pd();
	for (; n >= 2; n-=2, p+=2)
		s = _mm_add_pd(s, _mm_loadu_pd(p));
	a = _mm_shuffle_pd(s, s, 0x01);
	s = _mm_add_pd(s, a);
	double sum = _mm_cvtsd_f64(s);
#else
	double sum = 0;
#endif
	for (; n > 0; n--) sum += *p++;
	return sum;
}

// mul operation
static void fmul_f64(double *p, size_t n, double scalar)
{
#ifdef __SSE2__
	__m128d a = _mm_set_pd(scalar, scalar);
	for (; n >= 2; n-=2, p+=2)
	{
		__m128d v = _mm_mul_pd(_mm_loadu_pd(p), a);
		_mm_storeu_pd(p, v);
	}
#endif
	for (; n > 0; n--) (*p++) *= scalar;
}




// ===================================================================== //

/// set the internal variables
SEXP gpu_set_val(SEXP idx, SEXP val)
{
	switch (Rf_asInteger(idx))
	{
		case 0:
			gpu_f64_build_flag = LOGICAL(val)[0] == TRUE;
			gpu_f64_pred_flag  = LOGICAL(val)[1] == TRUE;
			break;
		case 1:
			kernel_build_prob = val; break;
		case 2:
			kernel_build_find_maxprob = val; break;
		case 3:
			kernel_build_sum_prob = val; break;
		case 4:
			kernel_build_clear = val; break;
		case 11:
			kernel_predict = val; break;
		case 12:
			kernel_predict_sumprob = val; break;
		case 13:
			kernel_predict_addprob = val; break;
	}
	return R_NilValue;
}


// ===================================================================== //

static inline void init_command(SEXP kernel)
{
	gpu_context = NULL;
	cl_int err = clGetKernelInfo(gpu_kernel, CL_KERNEL_CONTEXT,
		sizeof(gpu_context), &gpu_context, NULL);
	if (err != CL_SUCCESS || !gpu_context)
		throw err_text("Cannot obtain kernel context via clGetKernelInfo", err);

	// get device id from kernel R object
	cl_device_id device_id = getDeviceID(getAttrib(kernel, Rf_install("device")));
	// build a command
	gpu_commands = clCreateCommandQueue(gpu_context, device_id, 0, &err);
	if (!gpu_commands)
		throw err_text("Failed to create a command queue", err);
}

static inline void init_exp_log_memory(bool f64_flag)
{
	cl_int err;
	if (f64_flag)
	{
		GPU_CREATE_MEM(mem_exp_log_min_rare_freq,
			CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
			sizeof(EXP_LOG_MIN_RARE_FREQ), EXP_LOG_MIN_RARE_FREQ);
	} else {
		GPU_CREATE_MEM(mem_exp_log_min_rare_freq,
			CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
			sizeof(EXP_LOG_MIN_RARE_FREQ_f32), EXP_LOG_MIN_RARE_FREQ_f32);
	}
}

static inline void clear_prob_buffer(size_t size)
{
	cl_int err;
	int n = size >> 2;
	GPU_SETARG(gpu_kernel_clear, 0, n);
	size_t wdim = n / gpu_local_size_d1;
	if (n % gpu_local_size_d1) wdim++;
	wdim *= gpu_local_size_d1;
	GPU_RUN_KERNAL(gpu_kernel_clear, 1, &wdim, &gpu_local_size_d1);
}


// ===================================================================== //

// initialize the internal structure for building a model
void build_init(int nHLA, int nSample)
{
#ifdef HIBAG_ENABLE_TIMING
	memset(timing_array, 0, sizeof(timing_array));
	timing_array[TM_BUILD_TOTAL] = clock();
#endif

	cl_int err;
	gpu_kernel	= get_kernel(kernel_build_prob);
	gpu_kernel2 = get_kernel(kernel_build_find_maxprob);
	gpu_kernel3 = get_kernel(kernel_build_sum_prob);
	gpu_kernel_clear = get_kernel(kernel_build_clear);
	init_command(kernel_build_prob);

	// assign
	Num_HLA = nHLA;
	Num_Sample = nSample;
	const size_t size_hla = nHLA * (nHLA+1) >> 1;

	// initialize hla_map_index
	hla_map_index.resize(size_hla*2);
	int *p = &hla_map_index[0];
	for (int h1=0; h1 < nHLA; h1++)
		for (int h2=h1; h2 < nHLA; h2++)
			{ *p++ = h1; *p++ = h2; }


	// ====  allocate OpenCL buffers  ====

	// build_calc_prob -- exp_log_min_rare_freq
	init_exp_log_memory(gpu_f64_build_flag);

	// build_calc_prob -- pParam
	GPU_CREATE_MEM(mem_build_param, CL_MEM_READ_ONLY,
		sizeof(int) * (offset_build_param + 2*nSample), NULL);

	// build_calc_prob -- pHaplo
	mem_build_haplo_nmax = nSample * 8;
	const size_t msize_haplo = sizeof(THaplotype)*mem_build_haplo_nmax;
	GPU_CREATE_MEM(mem_haplo_list, CL_MEM_READ_ONLY, msize_haplo, NULL);

	// build_calc_prob -- pGeno
	GPU_CREATE_MEM(mem_snpgeno, CL_MEM_READ_ONLY, sizeof(TGenotype)*nSample, NULL);

	// build_find_maxprob -- out_idx
	GPU_CREATE_MEM(mem_build_output, CL_MEM_READ_WRITE, sizeof(double)*nSample*3, NULL);

	// build_calc_prob -- outProb
	msize_probbuf_each = size_hla * (gpu_f64_build_flag ? sizeof(double) : sizeof(float));
	for (mem_nmax_sample = nSample; mem_nmax_sample > 0; )
	{
		msize_probbuf_total = msize_probbuf_each * mem_nmax_sample;
		mem_prob_buffer = clCreateBuffer(gpu_context, CL_MEM_READ_WRITE,
			msize_probbuf_total, NULL, &err);
		if (mem_prob_buffer) break;
		mem_nmax_sample -= 256;
	}
	if (!mem_prob_buffer) 
		err_text("Unable to create buffer mem_prob_buffer", err);

	// arguments for gpu_kernel: build_calc_prob
	GPU_SETARG(gpu_kernel, 0, nHLA);
	GPU_SETARG(gpu_kernel, 1, mem_exp_log_min_rare_freq);
	GPU_SETARG(gpu_kernel, 2, mem_build_param);
	GPU_SETARG(gpu_kernel, 3, mem_haplo_list);
	GPU_SETARG(gpu_kernel, 4, mem_snpgeno);
	GPU_SETARG(gpu_kernel, 5, mem_prob_buffer);

	// arguments for gpu_kernel2: build_find_maxprob
	int sz_hla = size_hla;
	GPU_SETARG(gpu_kernel2, 0, sz_hla);
	GPU_SETARG(gpu_kernel2, 1, mem_prob_buffer);
	GPU_SETARG(gpu_kernel2, 2, mem_build_output);

	// arguments for gpu_kernel3: build_sum_prob
	GPU_SETARG(gpu_kernel3, 0, nHLA);
	GPU_SETARG(gpu_kernel3, 1, sz_hla);
	GPU_SETARG(gpu_kernel3, 2, mem_build_param);
	GPU_SETARG(gpu_kernel3, 3, mem_snpgeno);
	GPU_SETARG(gpu_kernel3, 4, mem_prob_buffer);
	GPU_SETARG(gpu_kernel3, 5, mem_build_output);

	// arguments for gpu_kernel_clear: clear_memory
	GPU_SETARG(gpu_kernel_clear, 1, mem_prob_buffer);
}


void build_done()
{
	hla_map_index.clear();
	GPU_FREE_MEM(mem_exp_log_min_rare_freq);
	GPU_FREE_MEM(mem_build_param);
	GPU_FREE_MEM(mem_haplo_list);
	GPU_FREE_MEM(mem_snpgeno);
	GPU_FREE_MEM(mem_build_output);
	GPU_FREE_MEM(mem_prob_buffer);
	GPU_FREE_COM(gpu_commands);

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
	cl_int err;
	void *ptr_buf;
	GPU_MAP_MEM(mem_build_param, ptr_buf,
		sizeof(int)*(offset_build_param + 2*Num_Sample));

	int *p_oob = (int*)ptr_buf + offset_build_param;
	int *p_ib  = p_oob + Num_Sample;
	num_oob = num_ib  = 0;
	for (int i=0; i < Num_Sample; i++)
	{
		if (oob_cnt[i] <= 0)
			p_oob[num_oob++] = i;
		else
			p_ib[num_ib++] = i;
	}

	GPU_UNMAP_MEM(mem_build_param, ptr_buf);
}

void build_set_haplo_geno(const THaplotype haplo[], int n_haplo,
	const TGenotype geno[], int n_snp)
{
	if (n_haplo > mem_build_haplo_nmax)
		throw "There are too many haplotypes out of the limit, please contact the package author.";

	cl_int err;
	wdim_n_haplo = num_haplo = n_haplo;
	if (wdim_n_haplo % gpu_local_size_d2)
		wdim_n_haplo = (wdim_n_haplo/gpu_local_size_d2 + 1)*gpu_local_size_d2;
	GPU_WRITE_MEM(mem_haplo_list, 0, sizeof(THaplotype)*n_haplo, (void*)haplo);

	num_snp = n_snp;
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
	if (num_oob <= 0) return 0;
	if (num_oob > mem_nmax_sample)
		throw "There are too many sample out of the limit of GPU memory, please contact the package author.";

	cl_int err;

	// initialize
	{
		HIBAG_TIMING(TM_BUILD_OOB_CLEAR)
		clear_prob_buffer(msize_probbuf_each * num_oob);
		int param[5] = { num_haplo, num_snp, 0, num_oob, offset_build_param };
		GPU_WRITE_MEM(mem_build_param, 0, sizeof(param), param);
	}

	// run OpenCL (calculating probabilities)
	{
		HIBAG_TIMING(TM_BUILD_OOB_PROB)
		size_t wdims[2] = { wdim_n_haplo, wdim_n_haplo };
		static const size_t local_size[2] = { gpu_local_size_d2, gpu_local_size_d2 };
		GPU_RUN_KERNAL(gpu_kernel, 2, wdims, local_size);
	}

	// find max index
	{
		HIBAG_TIMING(TM_BUILD_OOB_MAXIDX)
		size_t wdims[2] = { gpu_local_size_d1, num_oob };
		static const size_t local_size[2] = { gpu_local_size_d1, 1 };
		GPU_RUN_KERNAL(gpu_kernel2, 2, wdims, local_size);
	}

	// result
	int corrent_cnt = 0;

	void *ptr_geno, *ptr_index, *ptr_out;
	GPU_MAP_MEM(mem_snpgeno, ptr_geno, sizeof(TGenotype)*Num_Sample);
	GPU_MAP_MEM(mem_build_param, ptr_index, sizeof(int)*(offset_build_param + Num_Sample));
	GPU_MAP_MEM(mem_build_output, ptr_out, sizeof(int)*num_oob);

	TGenotype *pGeno = (TGenotype *)ptr_geno;
	int *pIdx = (int *)ptr_index + offset_build_param;
	int *pMaxI = (int *)ptr_out;
	THLAType hla;

	for (int i=0; i < num_oob; i++)
	{
		int k = pMaxI[i] << 1;
		if (k >= 0)
		{
			hla.Allele1 = hla_map_index[k];
			hla.Allele2 = hla_map_index[k+1];
		} else {
			hla.Allele1 = hla.Allele2 = NA_INTEGER;
		}
		corrent_cnt += compare(hla, pGeno[pIdx[i]].aux_hla_type);
	}

	GPU_UNMAP_MEM(mem_build_output, ptr_out);
	GPU_UNMAP_MEM(mem_build_param, ptr_index);
	GPU_UNMAP_MEM(mem_snpgeno, ptr_geno);

	return corrent_cnt;
}


double build_acc_ib()
{
	if (num_ib <= 0) return 0;
	if (num_ib > mem_nmax_sample)
		throw "There are too many sample out of the limit of GPU memory, please contact the package author.";

	cl_int err;

	// initialize
	{
		HIBAG_TIMING(TM_BUILD_IB_CLEAR)
		clear_prob_buffer(msize_probbuf_each * num_ib);
		int param[5] = { num_haplo, num_snp, 0, num_ib, offset_build_param+Num_Sample };
		GPU_WRITE_MEM(mem_build_param, 0, sizeof(param), param);
	}

	// run OpenCL (calculating probabilities)
	{
		HIBAG_TIMING(TM_BUILD_IB_PROB)
		size_t wdims[2] = { wdim_n_haplo, wdim_n_haplo };
		static const size_t local_size[2] = { gpu_local_size_d2, gpu_local_size_d2 };
		GPU_RUN_KERNAL(gpu_kernel, 2, wdims, local_size);
	}

	// result
	double LogLik = 0;

	// get log likelihood
	{
		HIBAG_TIMING(TM_BUILD_IB_LOGLIK)
		size_t wdims[2] = { gpu_local_size_d1, num_ib };
		static const size_t local_size[2] = { gpu_local_size_d1, 1 };
		GPU_RUN_KERNAL(gpu_kernel3, 2, wdims, local_size);
	}

	// sum of log likelihood
	void *ptr_out;
	GPU_MAP_MEM(mem_build_output, ptr_out, sizeof(double)*num_ib*3);

	if (gpu_f64_build_flag)
	{
		double *p = (double *)ptr_out;
		for (int i=0; i < num_ib; i++, p+=3)
		{
			if (p[0] > 0)
			{
				double v = p[2];
				if (v <= 0) v = 1e-128;
				LogLik += p[1] * log(v / p[0]);
			}
		}
	} else {
		float *p = (float *)ptr_out;
		for (int i=0; i < num_ib; i++, p+=3)
		{
			if (p[0] > 0)
			{
				double v = p[2];
				if (v <= 0) v = 1e-64;
				LogLik += p[1] * log(v / p[0]);
			}
		}
	}

	GPU_UNMAP_MEM(mem_build_output, ptr_out);

	return -2 * LogLik;
}



// ===================================================================== //

/// initialize the internal structure for predicting
void predict_init(int nHLA, int nClassifier, const THaplotype *const pHaplo[],
	const int nHaplo[])
{
	cl_int err;
	gpu_kernel	= get_kernel(kernel_predict);
	gpu_kernel2 = get_kernel(kernel_predict_sumprob);
	gpu_kernel3 = get_kernel(kernel_predict_addprob);
	init_command(kernel_predict);

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
	wdim_n_haplo = max_n_haplo;
	if (wdim_n_haplo % gpu_local_size_d2)
		wdim_n_haplo = (wdim_n_haplo/gpu_local_size_d2 + 1)*gpu_local_size_d2;

	// ====  allocate OpenCL buffers  ====

	// pred_calc_prob -- exp_log_min_rare_freq
	init_exp_log_memory(gpu_f64_pred_flag);

	// pred_calc_prob -- pHaplo
	const size_t msize_haplo = sizeof(THaplotype)*sum_n_haplo;
	GPU_CREATE_MEM(mem_haplo_list, CL_MEM_READ_ONLY, msize_haplo, NULL);
	void *ptr_haplo;
	GPU_MAP_MEM(mem_haplo_list, ptr_haplo, msize_haplo);
	THaplotype *p = (THaplotype *)ptr_haplo;
	for (int i=0; i < nClassifier; i++)
	{
		size_t m = nHaplo[i << 1];
		memcpy(p, pHaplo[i], sizeof(THaplotype)*m);
		p += m;
	}
	GPU_UNMAP_MEM(mem_haplo_list, ptr_haplo);

	// pred_calc_prob -- nHaplo
	GPU_CREATE_MEM(mem_pred_haplo_num, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
		sizeof(int)*2*nClassifier, (void*)nHaplo);

	// pred_calc_prob -- pGeno
	GPU_CREATE_MEM(mem_snpgeno, CL_MEM_READ_ONLY,
		sizeof(TGenotype)*nClassifier, NULL);

	// pred_calc_prob -- out_prob
	msize_probbuf_each = size_hla * (gpu_f64_pred_flag ? sizeof(double) : sizeof(float));
	msize_probbuf_total = msize_probbuf_each * nClassifier;
	GPU_CREATE_MEM(mem_prob_buffer, CL_MEM_READ_WRITE, msize_probbuf_total, NULL);

	// arguments for gpu_kernel, pred_calc_prob
	GPU_SETARG(gpu_kernel, 0, nHLA);
	GPU_SETARG(gpu_kernel, 1, nClassifier);
	GPU_SETARG(gpu_kernel, 2, mem_exp_log_min_rare_freq);
	GPU_SETARG(gpu_kernel, 3, mem_haplo_list);
	GPU_SETARG(gpu_kernel, 4, mem_pred_haplo_num);
	GPU_SETARG(gpu_kernel, 5, mem_snpgeno);
	GPU_SETARG(gpu_kernel, 6, mem_prob_buffer);

	// pred_calc_addprob -- weight
	GPU_CREATE_MEM(mem_pred_weight, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
		sizeof(double)*nClassifier, NULL);

	// arguments for gpu_kernel2, pred_calc_sumprob
	int sz_hla = size_hla;
	GPU_SETARG(gpu_kernel2, 0, sz_hla);
	GPU_SETARG(gpu_kernel2, 1, mem_prob_buffer);
	GPU_SETARG(gpu_kernel2, 2, mem_pred_weight);

	// arguments for gpu_kernel3, pred_calc_addprob
	GPU_SETARG(gpu_kernel3, 0, sz_hla);
	GPU_SETARG(gpu_kernel3, 1, nClassifier);
	GPU_SETARG(gpu_kernel3, 2, mem_pred_weight);
	GPU_SETARG(gpu_kernel3, 3, mem_prob_buffer);
	wdim_pred_addprob = size_hla;
	if (wdim_pred_addprob % gpu_local_size_d1)
		wdim_pred_addprob = (wdim_pred_addprob/gpu_local_size_d1 + 1) * gpu_local_size_d1;
}


/// finalize the structure for predicting
void predict_done()
{
	GPU_FREE_MEM(mem_exp_log_min_rare_freq);
	GPU_FREE_MEM(mem_haplo_list);
	GPU_FREE_MEM(mem_pred_haplo_num);
	GPU_FREE_MEM(mem_snpgeno);
	GPU_FREE_MEM(mem_prob_buffer);
	GPU_FREE_COM(gpu_commands);
}


/// average the posterior probabilities among classifiers for predicting
void predict_avg_prob(const TGenotype geno[], const double weight[],
	double out_prob[], double out_match[])
{
	const size_t num_size = Num_HLA * (Num_HLA + 1) >> 1;
	void *ptr_buf;

	// write to genotype buffer
	cl_int err;
	GPU_WRITE_MEM(mem_snpgeno, 0, sizeof(TGenotype)*Num_Classifier, geno);

	// initialize
	GPU_ZERO_FILL(mem_prob_buffer, msize_probbuf_total);

	// run OpenCL
	size_t wdims_k1[2] = { wdim_n_haplo, wdim_n_haplo };
	static const size_t local_size_k1[2] = { gpu_local_size_d2, gpu_local_size_d2 };
	GPU_RUN_KERNAL(gpu_kernel, 2, wdims_k1, local_size_k1);

/*
	// use host to calculate
	if (gpu_f64_pred_flag)
	{
	
	} else {
		// using double in hosts to improve precision
		GPU_MAP_MEM(mem_prob_buffer, ptr_buf, msize_probbuf_total);

		memset(out_prob, 0, sizeof(double)*num_size);
		const float *pb = (const float*)ptr_buf;
		double psum = 0;

		for (int i=0; i < Num_Classifier; i++)
		{
			double ss = 0;
			for (size_t k=0; k < num_size; k++)
				ss += pb[k];
			psum += ss;
			double w = (ss > 0) ? weight[i] / ss : 0;
			for (size_t k=0; k < num_size; k++)
				out_prob[k] += pb[k] * w;
			pb += num_size;
		}

		GPU_UNMAP_MEM(mem_prob_buffer, ptr_buf);

		*out_match = psum / Num_Classifier;
		fmul_f64(out_prob, num_size, 1 / get_sum_f64(out_prob, num_size));
		return;
	}
*/

	// sum up all probs per classifier
	size_t wdims_k2[2] = { gpu_local_size_d1, Num_Classifier };
	static const size_t local_size_k2[2] = { gpu_local_size_d1, 1 };
	GPU_RUN_KERNAL(gpu_kernel2, 2, wdims_k2, local_size_k2);

	// map to host memory
	GPU_MAP_MEM(mem_pred_weight, ptr_buf, sizeof(double)*Num_Classifier);
	double psum = 0;
	if (gpu_f64_pred_flag)
	{
		double *w = (double*)ptr_buf;
		for (int i=0; i < Num_Classifier; i++)
		{
			psum += w[i];
			w[i] = weight[i] / w[i];
			if (!R_FINITE(w[i])) w[i] = 0;
		}
	} else {
		float *w = (float*)ptr_buf;
		for (int i=0; i < Num_Classifier; i++)
		{
			psum += w[i];
			w[i] = weight[i] / w[i];
			if (!R_FINITE(w[i])) w[i] = 0;
		}
	}
	if (out_match)
		*out_match = psum / Num_Classifier;
	GPU_UNMAP_MEM(mem_pred_weight, ptr_buf);

	// sum up all probs among classifiers per HLA genotype
	size_t wdim = wdim_pred_addprob;
	GPU_RUN_KERNAL(gpu_kernel3, 1, &wdim, &gpu_local_size_d1);

	if (gpu_f64_pred_flag)
	{
		GPU_READ_MEM(mem_prob_buffer, sizeof(double)*num_size, out_prob);
		// normalize out_prob
		fmul_f64(out_prob, num_size, 1 / get_sum_f64(out_prob, num_size));
	} else {
		GPU_MAP_MEM(mem_prob_buffer, ptr_buf, sizeof(float)*num_size);
		const float *s = (const float*)ptr_buf;
		double *p=out_prob, sum=0;
		for (size_t n=num_size; n > 0; n--)
		{
			sum += *s; *p++ = *s++;
		}
		GPU_UNMAP_MEM(mem_prob_buffer, ptr_buf);

		// normalize out_prob
		fmul_f64(out_prob, num_size, 1 / sum);
	}
}



// ===================================================================== //

/// initialize GPU structure and return a pointer object
SEXP gpu_init_proc()
{
	const int n = 2 * HIBAG_MAXNUM_SNP_IN_CLASSIFIER;
	for (int i=0; i < n; i++)
		EXP_LOG_MIN_RARE_FREQ[i] = exp(i * log(MIN_RARE_FREQ));
	EXP_LOG_MIN_RARE_FREQ[0] = 1;
	for (int i=0; i < n; i++)
	{
		if (!R_finite(EXP_LOG_MIN_RARE_FREQ[i]))
			EXP_LOG_MIN_RARE_FREQ[i] = 0;
	}
	for (int i=0; i < n; i++)
		EXP_LOG_MIN_RARE_FREQ_f32[i] = double(1) * EXP_LOG_MIN_RARE_FREQ[i];

	GPU_Proc.build_init = build_init;
	GPU_Proc.build_done = build_done;
	GPU_Proc.build_set_bootstrap = build_set_bootstrap;
	GPU_Proc.build_set_haplo_geno = build_set_haplo_geno;
	GPU_Proc.build_acc_oob = build_acc_oob;
	GPU_Proc.build_acc_ib = build_acc_ib;

	GPU_Proc.predict_init = predict_init;
	GPU_Proc.predict_done = predict_done;
	GPU_Proc.predict_avg_prob = predict_avg_prob;

	return R_MakeExternalPtr(&GPU_Proc, R_NilValue, R_NilValue);
}

} // extern "C"
