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
#include <vector>

#ifdef __APPLE__
#   include <OpenCL/opencl.h>
#else
#   include <CL/opencl.h>
#endif

#define USE_RINTERNALS 1
#include <Rinternals.h>


// Streaming SIMD Extensions, SSE, SSE2, SSE4_2 (POPCNT)

#if (defined(__SSE__) && defined(__SSE2__))

#   include <xmmintrin.h>  // SSE
#   include <emmintrin.h>  // SSE2

#   if defined(__SSE4_2__) || defined(__POPCNT__)
#       define HIBAG_HARDWARE_POPCNT
#       include <nmmintrin.h>  // SSE4_2, for POPCNT
#   endif

#   define HIBAG_SIMD_OPTIMIZE_HAMMING_DISTANCE

#   define M128_I32_0(x)    _mm_cvtsi128_si32(x)
#   define M128_I32_1(x)    _mm_cvtsi128_si32(_mm_srli_si128(x, 4))
#   define M128_I32_2(x)    _mm_cvtsi128_si32(_mm_srli_si128(x, 8))
#   define M128_I32_3(x)    _mm_cvtsi128_si32(_mm_srli_si128(x, 12))
#   define M128_I64_0(x)    _mm_cvtsi128_si64(x) 
#   define M128_I64_1(x)    _mm_cvtsi128_si64(_mm_unpackhi_epi64(x, x))

#else

#   ifdef HIBAG_SIMD_OPTIMIZE_HAMMING_DISTANCE
#       undef HIBAG_SIMD_OPTIMIZE_HAMMING_DISTANCE
#   endif

#endif


// 32-bit or 64-bit registers

#ifdef __LP64__
#   define HIBAG_REG_BIT64
#else
#   ifdef HIBAG_REG_BIT64
#      undef HIBAG_REG_BIT64
#   endif
#endif


#define FREE_GPU_MEM(x)    if (x) { \
		clReleaseMemObject(x); x = NULL; \
	}

#define FREE_GPU_COM(x)    if (x) { \
		clReleaseCommandQueue(x); x = NULL; \
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
		double Frequency;
		/// old haplotype frequency
		double OldFreq;
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
		void (*predict_init)(int nHLA, int nClassifier, THaplotype *pHaplo[],
			int nHaplo[]);
		/// finalize the structure for predicting
		void (*predict_done)();
		/// average the posterior probabilities among classifiers for predicting
		void (*predict_avg_prob)(TGenotype *pGeno, const double weight[],
			double out_prob[]);
	} GPU_Proc;


	// GPU variables
	static int Num_HLA;
	static int Num_Classifier;

	static cl_context gpu_context = NULL;
	static cl_command_queue gpu_commands = NULL;
	static cl_kernel gpu_kernel = NULL;
	static bool gpu_f64_flag = false;

	// prediction for calculating the posterior probabilities
	static SEXP kernel_predict = NULL;
	static cl_mem mem_pred_haplo = NULL;
	static cl_mem mem_pred_haplo_num = NULL;
	static cl_mem mem_pred_tmp_prob = NULL;
	static cl_mem mem_pred_out_prob = NULL;
	static int wdim_pred = 0;
}



using namespace std;
using namespace HLA_LIB;

extern "C"
{

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
void predict_init(int nHLA, int nClassifier, THaplotype *pHaplo[], int nHaplo[])
{
	gpu_kernel = get_kernel(kernel_predict);
	gpu_context = NULL;
	if (clGetKernelInfo(gpu_kernel, CL_KERNEL_CONTEXT,
			sizeof(gpu_context), &gpu_context, NULL) != CL_SUCCESS || !gpu_context)
		throw "cannot obtain kernel context via clGetKernelInfo";

	Num_HLA = nHLA;
	int size_hla = nHLA * (nHLA+1) / 2;
	Num_Classifier = nClassifier;

	// the number of haplotypes among all classifiers in total
	size_t sum_n_haplo = 0, max_n_haplo=0;
	for (int i=0; i < nClassifier; i++)
	{
		sum_n_haplo += nHaplo[i];
		if (nHaplo[i] > max_n_haplo)
			max_n_haplo = nHaplo[i];
	}
	wdim_pred = max_n_haplo;
	// linear storage of all haplotypes
	vector<THaplotype> buf_haplo(sum_n_haplo);
	THaplotype *p = &buf_haplo[0];
	for (int i=0; i < nClassifier; i++)
	{
		memcpy(p, pHaplo[i], sizeof(THaplotype)*nHaplo[i]);
		p += nHaplo[i];
	}

	// allocate OpenCL buffers
	cl_int err;
	mem_pred_haplo = clCreateBuffer(gpu_context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(THaplotype)*sum_n_haplo,
		&buf_haplo[0], &err);
	if (!mem_pred_haplo)
		throw "Unable to create buffer.";
	mem_pred_haplo_num = clCreateBuffer(gpu_context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int)*nClassifier,
		nHaplo, &err);
	if (!mem_pred_haplo_num)
		throw "Unable to create buffer.";
	mem_pred_tmp_prob = clCreateBuffer(gpu_context,
		CL_MEM_READ_WRITE, sizeof(double)*size_hla, NULL, &err);
	if (!mem_pred_tmp_prob)
		throw "Unable to create buffer.";
	mem_pred_out_prob = clCreateBuffer(gpu_context,
		CL_MEM_READ_WRITE, sizeof(double)*size_hla, NULL, &err);
	if (!mem_pred_out_prob)
		throw "Unable to create buffer.";

	// arguments
    if (clSetKernelArg(gpu_kernel, 0, sizeof(cl_mem), &mem_pred_out_prob) != CL_SUCCESS)
    	throw "Failed to set kernel argument (0).";
    if (clSetKernelArg(gpu_kernel, 1, sizeof(int), &nHLA) != CL_SUCCESS)
    	throw "Failed to set kernel argument (1).";
    if (clSetKernelArg(gpu_kernel, 2, sizeof(int), &nClassifier) != CL_SUCCESS)
    	throw "Failed to set kernel argument (2).";
    if (clSetKernelArg(gpu_kernel, 3, sizeof(cl_mem), &mem_pred_haplo) != CL_SUCCESS)
    	throw "Failed to set kernel argument (3).";
    if (clSetKernelArg(gpu_kernel, 4, sizeof(cl_mem), &mem_pred_haplo_num) != CL_SUCCESS)
    	throw "Failed to set kernel argument (4).";
    if (clSetKernelArg(gpu_kernel, 5, sizeof(cl_mem), &mem_pred_tmp_prob) != CL_SUCCESS)
    	throw "Failed to set kernel argument (5).";

	// get device id from kernel R object
	cl_device_id device_id = getDeviceID(getAttrib(kernel_predict,
		Rf_install("device")));
	// build a command
	gpu_commands = clCreateCommandQueue(gpu_context, device_id, 0, &err);
}


/// finalize the structure for predicting
void predict_done()
{
	FREE_GPU_MEM(mem_pred_haplo);
	FREE_GPU_MEM(mem_pred_haplo_num);
	FREE_GPU_MEM(mem_pred_tmp_prob);
	FREE_GPU_MEM(mem_pred_out_prob);
	FREE_GPU_COM(gpu_commands);
}


/// average the posterior probabilities among classifiers for predicting
void predict_avg_prob(TGenotype *pGeno, const double weight[],
	double out_prob[])
{
	// run
	size_t wdims[2] = { wdim_pred, wdim_pred };
	if (clEnqueueNDRangeKernel(gpu_commands, gpu_kernel, 2, NULL, wdims, NULL, 0, NULL, NULL) != CL_SUCCESS)
		throw "Failed to run clEnqueueNDRangeKernel().";
	clFinish(gpu_commands);

	const size_t num = Num_HLA * (Num_HLA + 1) / 2;

	// output
	if (clEnqueueReadBuffer(gpu_commands, mem_pred_out_prob, CL_TRUE, 0,
			sizeof(double) * num,
			out_prob, 0, NULL, NULL) != CL_SUCCESS)
		throw "Failed to read memory buffer.";

	// if no support of double floating precision
	if (!gpu_f64_flag)
	{
		float *s = (float*)out_prob + num;
		double *p = out_prob + num;
		for (size_t n=num; n > 0; n--) *(--p) = *(--s);
	}
}


/// initialize GPU structure and return a pointer object
SEXP init_gpu_proc()
{
	GPU_Proc.predict_init = predict_init;
	GPU_Proc.predict_done = predict_done;
	GPU_Proc.predict_avg_prob = predict_avg_prob;
	return R_MakeExternalPtr(&GPU_Proc, R_NilValue, R_NilValue);
}

} // extern "C"
