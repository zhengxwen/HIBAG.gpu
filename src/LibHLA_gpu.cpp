// ===============================================================
//
// HIBAG.gpu R package (GPU-based implementation for the HIBAG package)
// Copyright (C) 2017   Xiuwen Zheng (zhengx@u.washington.edu)
// All rights reserved.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.


#include <stdint.h>
#include <string.h>
#include <cstdlib>
#include <cmath>
#include <vector>

#ifdef __APPLE__
#   include <OpenCL/opencl.h>
#else
#   include <CL/opencl.h>
#endif

#define USE_RINTERNALS 1
#include <Rinternals.h>


// Streaming SIMD Extensions, SSE, SSE2, SSE4_2 (POPCNT)

#ifdef __SSE__
#   include <xmmintrin.h>  // SSE
#endif

#ifdef __SSE2__
#   include <emmintrin.h>  // SSE2
#endif


// 32-bit or 64-bit registers

#ifdef __LP64__
#   define HIBAG_REG_BIT64
#else
#   ifdef HIBAG_REG_BIT64
#      undef HIBAG_REG_BIT64
#   endif
#endif


#define GPU_CREATE_MEM(x, flag, size, ptr)    \
	x = clCreateBuffer(gpu_context, flag, size, ptr, &err); \
	if (!x) throw err_text("Unable to create buffer " #x, err);

#define GPU_MAP_MEM(x, ptr, size)    \
	ptr = clEnqueueMapBuffer(gpu_commands, x, CL_TRUE, \
		CL_MAP_READ | CL_MAP_WRITE, 0, size, 0, NULL, NULL, &err); \
	if (!ptr) throw err_text("Unable to map buffer to host memory " #x, err);

#define GPU_UNMAP_MEM(x, ptr)    \
	err = clEnqueueUnmapMemObject(gpu_commands, x, ptr, 0, NULL, NULL); \
	if (err != CL_SUCCESS) \
		throw err_text("Failed to unmap memory buffer " #x, err); \
	ptr = NULL;

#define GPU_READ_MEM(x, size, ptr)    \
	err = clEnqueueReadBuffer(gpu_commands, x, CL_TRUE, 0, size, ptr, 0, NULL, NULL); \
	if (err != CL_SUCCESS) \
		throw err_text("Failed to read memory buffer " #x, err);

#define GPU_WRITE_MEM(x, offset, size, ptr)    \
	err = clEnqueueWriteBuffer(gpu_commands, x, CL_TRUE, offset, size, ptr, 0, NULL, NULL); \
	if (err != CL_SUCCESS) \
		throw err_text("Failed to write memory buffer " #x, err);

#define GPU_SETARG(i, x)    \
	err = clSetKernelArg(gpu_kernel, i, sizeof(x), &x); \
	if (err != CL_SUCCESS) \
		throw err_text("Failed to set kernel argument (" #i ")", err);

#define GPU_RUN_KERNAL(k, ndim, wdims, lsize)    \
	err = clEnqueueNDRangeKernel(gpu_commands, k, ndim, NULL, wdims, lsize, 0, NULL, NULL); \
	if (err != CL_SUCCESS) \
		throw err_text("Failed to run clEnqueueNDRangeKernel() with " #k, err); \
	err = clFinish(gpu_commands); \
	if (err != CL_SUCCESS) \
		throw err_text("Failed to run clFinish() with " #k, err);

#define GPU_FREE_MEM(x)    if (x) { \
		if (clReleaseMemObject(x) != CL_SUCCESS) \
			throw "Failed to free memory buffer"; \
		x = NULL; \
	}

#define GPU_FREE_COM(x)    if (x) { \
		if (clReleaseCommandQueue(x) != CL_SUCCESS) \
			throw "Failed to release command queue."; \
		x = NULL; \
	}



namespace HLA_LIB
{
	using namespace std;

	/// Define unsigned integers
	typedef uint8_t     UINT8;

	/// The max number of SNP markers in an individual classifier.
	//  Don't modify this value since the code is optimized for this value!!!
	const size_t HIBAG_MAXNUM_SNP_IN_CLASSIFIER = 128;

	/// The max number of UTYPE for packed SNP genotypes.
	const size_t HIBAG_PACKED_UTYPE_MAXNUM =
		HIBAG_MAXNUM_SNP_IN_CLASSIFIER / (8*sizeof(UINT8));


	// ===================================================================== //
	// ========                     Description                     ========
	//
	// Packed SNP storage strategy is used for faster matching
	//
	// HLA allele: start from 0
	//
	// THaplotype: packed SNP alleles (little endianness):
	//     (s8 s7 s6 s5 s4 s3 s2 s1)
	//     the 1st allele: (s1), the 2nd allele: (s2), ...
	//     SNP allele: 0 (B allele), 1 (A allele)
	//
	// TGenotype: packed SNP genotype (little endianness):
	//     array_1 = (s1_8 s1_7 s1_6 s1_5 s1_4 s1_3 s1_2 s1_1),
	//     array_2 = (s2_8 s2_7 s2_6 s2_5 s2_4 s2_3 s2_2 s2_1),
	//     array_3 = (s3_8 s3_7 s3_6 s3_5 s3_4 s3_3 s3_2 s3_1)
	//     the 1st genotype: (s1_1 s2_1 s3_1),
	//     the 2nd genotype: (s1_1 s2_1 s3_1), ...
	//     SNP genotype: 0 (BB) -- (s1_1=0 s2_1=0 s3_1=1),
	//                   1 (AB) -- (s1_1=1 s2_1=0 s3_1=1),
	//                   2 (AA) -- (s1_1=1 s2_1=1 s3_1=1),
	//                   -1 or other value (missing)
	//                          -- (s1_1=0 s2_1=0 s3_1=0)
	//
	// ========                                                     ========
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
			double OldFreq;  /// old haplotype frequency
			struct type_aux2 {
				float Freq_f32;  /// 32-bit haplotype frequency
				int HLA_allele;  /// the associated HLA allele
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



	const int gpu_local_size = 4;

	// GPU variables
	static int Num_HLA;
	static int Num_Classifier;

	static cl_context gpu_context = NULL;
	static cl_command_queue gpu_commands = NULL;
	static cl_kernel gpu_kernel = NULL;
	static bool gpu_f64_flag = false;


	// prediction for calculating the posterior probabilities
	static SEXP kernel_predict = NULL;

	// the buffer of numeric values and parameters
	// int[0] -- the index of classifier
	// int[1] -- the start of haplotype list for the current classifier
	// double[] -- sizeof(EXP_LOG_MIN_RARE_FREQ)
	// double[] -- nHLA*(nHLA+1)/2, tmp_prob
	// double[] -- nHLA*(nHLA+1)/2, out_prob
	// double   -- temporary
	static cl_mem mem_pred_buf_param = NULL;
	static size_t memsize_buf_param = 0;
	static size_t memsize_prob = 0;
	const static size_t offset_pred_buf_exp_log = sizeof(int)*2;
	static size_t offset_tmp_prob = 0;
	static size_t offset_out_prob = 0;


	// haplotypes of all classifiers
	static cl_mem mem_pred_haplo = NULL;
	// num of haplotypes and SNPs for each classifier
	static cl_mem mem_pred_haplo_num = NULL;
	// SNP genotypes
	static cl_mem mem_pred_snpgeno = NULL;

	static int wdim_pred = 0;
}



using namespace std;
using namespace HLA_LIB;

extern "C"
{

/// Frequency Calculation
#define FREQ_MUTANT(p, cnt)    ((p) * EXP_LOG_MIN_RARE_FREQ[cnt]);
/// the minimum rare frequency to store haplotypes
#define MIN_RARE_FREQ    1e-5

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

// add mul operation
static void faddmul_f32(float *p, const float *s, size_t n, float scalar)
{
#ifdef __SSE__
	__m128 a = _mm_set_ps1(scalar);
	for (; n >= 4; n-=4, s+=4, p+=4)
	{
		__m128 s4 = _mm_mul_ps(_mm_loadu_ps(s), a);
		__m128 p4 = _mm_add_ps(s4, _mm_loadu_ps(p));
		_mm_storeu_ps(p, p4);
	}
#endif
	for (; n > 0; n--)
		(*p++) += (*s++) * scalar;
}

// add mul operation
static void faddmul_f64(double *p, const double *s, size_t n, double scalar)
{
#ifdef __SSE2__
	__m128d a = _mm_set_pd(scalar, scalar);
	for (; n >= 2; n-=2, s+=2, p+=2)
	{
		__m128d s2 = _mm_mul_pd(_mm_loadu_pd(s), a);
		__m128d p2 = _mm_add_pd(s2, _mm_loadu_pd(p));
		_mm_storeu_pd(p, p2);
	}
#endif
	for (; n > 0; n--)
		(*p++) += (*s++) * scalar;
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
	}
	return R_NilValue;
}


/// initialize the internal structure for predicting
void predict_init(int nHLA, int nClassifier, const THaplotype *const pHaplo[],
	const int nHaplo[])
{
	cl_int err;
	gpu_kernel = get_kernel(kernel_predict);
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
	if (wdim_pred % gpu_local_size)
		wdim_pred = (wdim_pred/gpu_local_size + 1)*gpu_local_size;
	Rprintf("global_size: %d\n", (int)wdim_pred);

	// allocate OpenCL buffers
	memsize_buf_param = sizeof(int)*2 + sizeof(EXP_LOG_MIN_RARE_FREQ) +
		sizeof(double)*size_hla*2;
	memsize_prob = size_hla * (gpu_f64_flag ? sizeof(double) : sizeof(float));
	offset_tmp_prob = offset_pred_buf_exp_log + (gpu_f64_flag ?
		sizeof(EXP_LOG_MIN_RARE_FREQ) : sizeof(EXP_LOG_MIN_RARE_FREQ_f32));
	offset_out_prob = offset_tmp_prob + memsize_prob;

	GPU_CREATE_MEM(mem_pred_buf_param, CL_MEM_READ_WRITE, memsize_buf_param, NULL);
	if (gpu_f64_flag)
	{
		GPU_WRITE_MEM(mem_pred_buf_param, offset_pred_buf_exp_log,
			sizeof(EXP_LOG_MIN_RARE_FREQ), EXP_LOG_MIN_RARE_FREQ);
	} else {
		GPU_WRITE_MEM(mem_pred_buf_param, offset_pred_buf_exp_log,
			sizeof(EXP_LOG_MIN_RARE_FREQ_f32), EXP_LOG_MIN_RARE_FREQ_f32);
	}
	//
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
	//
	GPU_CREATE_MEM(mem_pred_haplo_num, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
		sizeof(int)*nClassifier*2, (void*)nHaplo);
	GPU_CREATE_MEM(mem_pred_snpgeno, CL_MEM_READ_ONLY,
		sizeof(TGenotype)*nClassifier, NULL);

	// arguments
	GPU_SETARG(0, nHLA);
	GPU_SETARG(1, mem_pred_buf_param);
	GPU_SETARG(2, mem_pred_haplo);
	GPU_SETARG(3, mem_pred_haplo_num);
	GPU_SETARG(4, mem_pred_snpgeno);
}


/// finalize the structure for predicting
void predict_done()
{
	GPU_FREE_MEM(mem_pred_buf_param);
	GPU_FREE_MEM(mem_pred_haplo);
	GPU_FREE_MEM(mem_pred_haplo_num);
	GPU_FREE_MEM(mem_pred_snpgeno);
	GPU_FREE_COM(gpu_commands);
}


/// average the posterior probabilities among classifiers for predicting
void predict_avg_prob(const int nHaplo[], const TGenotype geno[],
	const double weight[], double out_prob[])
{
	const size_t num_size = Num_HLA * (Num_HLA + 1) >> 1;

	// write to genotype buffer
	cl_int err;
	GPU_WRITE_MEM(mem_pred_snpgeno, 0, sizeof(TGenotype)*Num_Classifier, geno);

	// memory mapping
	void *ptr_buf = NULL;
	GPU_MAP_MEM(mem_pred_buf_param, ptr_buf, memsize_buf_param);

	// initialize
	((int*)ptr_buf)[1] = 0;  // start_haplo
	memset((char*)ptr_buf + offset_out_prob, 0, memsize_prob);

	size_t wdims[2] = { wdim_pred, wdim_pred };
	static size_t local_size[2] = { gpu_local_size, gpu_local_size };

	// run
	for (int i=0; i < Num_Classifier; i++)
	{
		// initialize
		int &i_classifier = ((int*)ptr_buf)[0];
		i_classifier = i;
		memset((char*)ptr_buf + offset_tmp_prob, 0, memsize_prob);
		// update GPU memory
		GPU_UNMAP_MEM(mem_pred_buf_param, ptr_buf);

		// run OpenCL
		GPU_RUN_KERNAL(gpu_kernel, 2, wdims, local_size);

		// map to host memory
		GPU_MAP_MEM(mem_pred_buf_param, ptr_buf, memsize_buf_param);
		// update
		int &start_haplo = ((int*)ptr_buf)[1];
		start_haplo += nHaplo[2*i];
		if (gpu_f64_flag)
		{
			double *s = (double*)((char*)ptr_buf + offset_tmp_prob);
			double scalar = weight[i] / get_sum_f64(s, num_size);
			faddmul_f64((double*)((char*)ptr_buf + offset_out_prob), s,
				num_size, scalar);
		} else {
			float *s = (float*)((char*)ptr_buf + offset_tmp_prob);
			float scalar = weight[i] / get_sum_f32(s, num_size);
			faddmul_f32((float*)((char*)ptr_buf + offset_out_prob), s,
				num_size, scalar);
		}
	}

	// if no support of double floating precision
	if (gpu_f64_flag)
	{
		double *s = (double*)((char*)ptr_buf + offset_out_prob);
		memcpy(out_prob, s, num_size * sizeof(double));
	} else {
		float *s = (float*)((char*)ptr_buf + offset_out_prob);
		float scalar = 1 / get_sum_f32(s, num_size);
		double *p = out_prob;
		for (size_t n=num_size; n > 0; n--) *p++ = (*s++) * scalar;
	}

	GPU_UNMAP_MEM(mem_pred_buf_param, ptr_buf);
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
