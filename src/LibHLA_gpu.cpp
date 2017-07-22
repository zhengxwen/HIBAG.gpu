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


// 32-bit or 64-bit registers

#ifdef __LP64__
#	define HIBAG_REG_BIT64
#else
#	ifdef HIBAG_REG_BIT64
#	   undef HIBAG_REG_BIT64
#	endif
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
		size_t zero = 0; \
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
		void (*build_acc_init)(int nHLA, THaplotype pHaplo[], int nHaplo,
			TGenotype pGeno[], int nGeno);
		/// finalize the structure for building a model
		void (*build_acc_done)();
		/// calculate the out-of-bag accuracy (the number of correct alleles)
		int (*build_acc_oob)();
		/// calculate the in-bag log likelihood
		double (*build_acc_ib)();

		/// initialize the internal structure for predicting
		void (*predict_init)(int nHLA, int nClassifier,
			const THaplotype *const pHaplo[], const int nHaplo[]);
		/// finalize the structure for predicting
		void (*predict_done)();
		/// average the posterior probabilities among classifiers for predicting
		void (*predict_avg_prob)(const int nHaplo[], const TGenotype geno[],
			const double weight[], double out_prob[]);
	} GPU_Proc;



	const size_t gpu_local_size_d1 = 32;
	const size_t gpu_local_size_d2 = 4;

	// GPU variables
	static int Num_HLA;
	static int Num_Classifier;

	static cl_context gpu_context = NULL;
	static cl_command_queue gpu_commands = NULL;
	static cl_kernel gpu_kernel	 = NULL;
	static cl_kernel gpu_kernel2 = NULL;
	static cl_kernel gpu_kernel3 = NULL;
	static bool gpu_f64_flag = false;

	// haplotypes of all classifiers
	static cl_mem mem_exp_log_min_rare_freq = NULL;


	// ===================================================================== //
	// prediction for calculating the posterior probabilities

	static SEXP kernel_predict = NULL;
	static SEXP kernel_predict_sumprob = NULL;
	static SEXP kernel_predict_addprob = NULL;

	// haplotypes of all classifiers
	static cl_mem mem_pred_haplo = NULL;
	// num of haplotypes and SNPs for each classifier
	static cl_mem mem_pred_haplo_num = NULL;
	// SNP genotypes
	static cl_mem mem_pred_snpgeno = NULL;

	// the buffer of numeric values
	// double[nHLA*(nHLA+1)/2][] -- tmp_prob for each classifier
	static cl_mem mem_pred_probbuf = NULL;
	static size_t memsize_buf_param = 0;
	static size_t memsize_prob = 0;

	// classifier weight
	static cl_mem mem_pred_weight = NULL;

	static int wdim_pred = 0;
	static int wdim_pred_addprob = 0;
}



using namespace std;
using namespace HLA_LIB;

extern "C"
{

/// Frequency Calculation
#define FREQ_MUTANT(p, cnt)	   ((p) * EXP_LOG_MIN_RARE_FREQ[cnt]);
/// the minimum rare frequency to store haplotypes
#define MIN_RARE_FREQ	 1e-5

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


SEXP ocl_build_kernel(SEXP device, SEXP k_name, SEXP code, SEXP prec)
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


// get the sum of float array
static float get_sum_f32(const float *p, size_t n)
{
#ifdef __SSE__
	__m128 a, s = _mm_setzero_ps();
	for (; n >= 4; n-=4, p+=4)
		s = _mm_add_ps(s, _mm_loadu_ps(p));
	a = _mm_shuffle_ps(s, s, _MM_SHUFFLE(1,0,3,2));
	s = _mm_add_ps(s, a);
	a = _mm_shuffle_ps(s, s, _MM_SHUFFLE(0,0,0,1));
	s = _mm_add_ps(s, a);
	float sum = _mm_cvtss_f32(s);
#else
	float sum = 0;
#endif
	for (; n > 0; n--) sum += *p++;
	return sum;
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
static void fmul_f32(float *p, size_t n, float scalar)
{
#ifdef __SSE2__
	__m128 a = _mm_set_ps1(scalar);
	for (; n >= 4; n-=4, p+=4)
	{
		__m128 v = _mm_mul_ps(_mm_loadu_ps(p), a);
		_mm_storeu_ps(p, v);
	}
#endif
	for (; n > 0; n--) (*p++) *= scalar;
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
SEXP set_gpu_val(SEXP idx, SEXP val)
{
	switch (Rf_asInteger(idx))
	{
		case 0:
			gpu_f64_flag = Rf_asLogical(val) == TRUE; break;
		case 1:
			kernel_predict = val; break;
		case 2:
			kernel_predict_sumprob = val; break;
		case 3:
			kernel_predict_addprob = val; break;
	}
	return R_NilValue;
}


/// initialize the internal structure for predicting
void predict_init(int nHLA, int nClassifier, const THaplotype *const pHaplo[],
	const int nHaplo[])
{
	cl_int err;
	gpu_kernel	= get_kernel(kernel_predict);
	gpu_kernel2 = get_kernel(kernel_predict_sumprob);
	gpu_kernel3 = get_kernel(kernel_predict_addprob);
	gpu_context = NULL;
	if (clGetKernelInfo(gpu_kernel, CL_KERNEL_CONTEXT,
			sizeof(gpu_context), &gpu_context, NULL) != CL_SUCCESS || !gpu_context)
		throw "Cannot obtain kernel context via clGetKernelInfo";

	// get device id from kernel R object
	cl_device_id device_id = getDeviceID(getAttrib(kernel_predict,
		Rf_install("device")));
	// build a command
	gpu_commands = clCreateCommandQueue(gpu_context, device_id, 0, &err);
	if (!gpu_commands)
		throw err_text("Failed to create a command queue", err);

	// assign
	Num_HLA = nHLA;
	size_t size_hla = nHLA * (nHLA+1) >> 1;
	Num_Classifier = nClassifier;

	// the number of haplotypes among all classifiers in total
	size_t sum_n_haplo = 0, max_n_haplo=0;
	for (int i=0; i < nClassifier; i++)
	{
		size_t m = nHaplo[i << 1];
		sum_n_haplo += m;
		if (m > max_n_haplo) max_n_haplo = m;
	}
	wdim_pred = max_n_haplo;
	if (wdim_pred % gpu_local_size_d2)
		wdim_pred = (wdim_pred/gpu_local_size_d2 + 1)*gpu_local_size_d2;

	// allocate OpenCL buffers

	// pred_calc_prob -- exp_log_min_rare_freq
	if (gpu_f64_flag)
	{
		GPU_CREATE_MEM(mem_exp_log_min_rare_freq, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
			sizeof(EXP_LOG_MIN_RARE_FREQ), EXP_LOG_MIN_RARE_FREQ);
	} else {
		GPU_CREATE_MEM(mem_exp_log_min_rare_freq, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
			sizeof(EXP_LOG_MIN_RARE_FREQ_f32), EXP_LOG_MIN_RARE_FREQ_f32);
	}

	// pred_calc_prob -- pHaplo
	const size_t msize_haplo = sizeof(THaplotype)*sum_n_haplo;
	GPU_CREATE_MEM(mem_pred_haplo, CL_MEM_READ_ONLY, msize_haplo, NULL);
	void *ptr_haplo;
	GPU_MAP_MEM(mem_pred_haplo, ptr_haplo, msize_haplo);
	THaplotype *p = (THaplotype *)ptr_haplo;
	for (int i=0; i < nClassifier; i++)
	{
		size_t m = nHaplo[i << 1];
		memcpy(p, pHaplo[i], sizeof(THaplotype)*m);
		p += m;
	}
	GPU_UNMAP_MEM(mem_pred_haplo, ptr_haplo);

	// pred_calc_prob -- nHaplo
	GPU_CREATE_MEM(mem_pred_haplo_num, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
		sizeof(int)*nClassifier*2, (void*)nHaplo);

	// pred_calc_prob -- pGeno
	GPU_CREATE_MEM(mem_pred_snpgeno, CL_MEM_READ_ONLY,
		sizeof(TGenotype)*nClassifier, NULL);

	// pred_calc_prob -- out_prob
	memsize_prob = size_hla * (gpu_f64_flag ? sizeof(double) : sizeof(float));
	memsize_buf_param = memsize_prob * nClassifier;
	GPU_CREATE_MEM(mem_pred_probbuf, CL_MEM_READ_WRITE, memsize_buf_param, NULL);

	// arguments for gpu_kernel
	GPU_SETARG(gpu_kernel, 0, nHLA);
	GPU_SETARG(gpu_kernel, 1, nClassifier);
	GPU_SETARG(gpu_kernel, 2, mem_exp_log_min_rare_freq);
	GPU_SETARG(gpu_kernel, 3, mem_pred_haplo);
	GPU_SETARG(gpu_kernel, 4, mem_pred_haplo_num);
	GPU_SETARG(gpu_kernel, 5, mem_pred_snpgeno);
	GPU_SETARG(gpu_kernel, 6, mem_pred_probbuf);

	// pred_calc_addprob -- weight
	GPU_CREATE_MEM(mem_pred_weight, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
		sizeof(double)*nClassifier, NULL);

	// arguments for gpu_kernel2
	int sz_hla = size_hla;
	GPU_SETARG(gpu_kernel2, 0, sz_hla);
	int sz_per_local = sz_hla / gpu_local_size_d1;
	if (sz_hla % gpu_local_size_d1) sz_per_local++;
	GPU_SETARG(gpu_kernel2, 1, sz_per_local);
	GPU_SETARG(gpu_kernel2, 2, nClassifier);
	GPU_SETARG(gpu_kernel2, 3, mem_pred_probbuf);
	GPU_SETARG(gpu_kernel2, 4, mem_pred_weight);

	// arguments for gpu_kernel3
	GPU_SETARG(gpu_kernel3, 0, sz_hla);
	GPU_SETARG(gpu_kernel3, 1, nClassifier);
	GPU_SETARG(gpu_kernel3, 2, mem_pred_weight);
	GPU_SETARG(gpu_kernel3, 3, mem_pred_probbuf);
	wdim_pred_addprob = size_hla;
	if (wdim_pred_addprob % gpu_local_size_d1)
		wdim_pred_addprob = (wdim_pred/gpu_local_size_d1 + 1)*gpu_local_size_d1;
}


/// finalize the structure for predicting
void predict_done()
{
	GPU_FREE_MEM(mem_exp_log_min_rare_freq);
	GPU_FREE_MEM(mem_pred_haplo);
	GPU_FREE_MEM(mem_pred_haplo_num);
	GPU_FREE_MEM(mem_pred_snpgeno);
	GPU_FREE_MEM(mem_pred_probbuf);
	GPU_FREE_COM(gpu_commands);
}


/// average the posterior probabilities among classifiers for predicting
void predict_avg_prob(const int nHaplo[], const TGenotype geno[],
	const double weight[], double out_prob[])
{
	const size_t num_size = Num_HLA * (Num_HLA + 1) >> 1;
	void *ptr_buf;

	// write to genotype buffer
	cl_int err;
	GPU_WRITE_MEM(mem_pred_snpgeno, 0, sizeof(TGenotype)*Num_Classifier, geno);

	// initialize
	GPU_ZERO_FILL(mem_pred_probbuf, memsize_buf_param);

	// run OpenCL
	size_t wdims_k1[2] = { wdim_pred, wdim_pred };
	static size_t local_size_k1[2] = { gpu_local_size_d2, gpu_local_size_d2 };
	GPU_RUN_KERNAL(gpu_kernel, 2, wdims_k1, local_size_k1);

	// sum up all probs per classifier
	size_t wdims_k2[2] = { gpu_local_size_d1, Num_Classifier };
	static size_t local_size_k2[2] = { gpu_local_size_d1, 1 };
	GPU_RUN_KERNAL(gpu_kernel2, 2, wdims_k2, local_size_k2);


	// map to host memory
	GPU_MAP_MEM(mem_pred_weight, ptr_buf, sizeof(double)*Num_Classifier);
	if (gpu_f64_flag)
	{
		double *s = (double*)ptr_buf;
		for (int i=0; i < Num_Classifier; i++)
		{
			double scalar = weight[i] / get_sum_f64(s, num_size);
			double *p = out_prob;
			for (size_t n=num_size; n > 0; n--)
				*p++ += scalar * (*s++);
		}
		// normalize out_prob
		fmul_f64(out_prob, num_size, 1 / get_sum_f64(out_prob, num_size));

	} else {
		float *w = (float*)ptr_buf;
		for (int i=0; i < Num_Classifier; i++)
			w[i] = weight[i] / w[i];
	}
	GPU_UNMAP_MEM(mem_pred_weight, ptr_buf);

	// sum up all probs among classifiers per HLA genotype
	size_t wdim = wdim_pred_addprob;
	GPU_RUN_KERNAL(gpu_kernel3, 1, &wdim, &gpu_local_size_d1);

	if (gpu_f64_flag)
	{
	} else {
		GPU_MAP_MEM(mem_pred_probbuf, ptr_buf, sizeof(float)*num_size);
		const float *s = (const float*)ptr_buf;
		double *p = out_prob;
		for (size_t n=num_size; n > 0; n--) *p++ = *s++;
		GPU_UNMAP_MEM(mem_pred_probbuf, ptr_buf);
	}

	// normalize out_prob
	fmul_f64(out_prob, num_size, 1 / get_sum_f64(out_prob, num_size));
}


/// initialize GPU structure and return a pointer object
SEXP init_gpu_proc()
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
		EXP_LOG_MIN_RARE_FREQ_f32[i] = EXP_LOG_MIN_RARE_FREQ[i];

	GPU_Proc.predict_init = predict_init;
	GPU_Proc.predict_done = predict_done;
	GPU_Proc.predict_avg_prob = predict_avg_prob;
	return R_MakeExternalPtr(&GPU_Proc, R_NilValue, R_NilValue);
}

} // extern "C"
