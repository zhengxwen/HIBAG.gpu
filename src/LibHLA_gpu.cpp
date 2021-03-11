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


#include "LibOpenCL.h"

// Defined in HIBAG/install/LibHLA_ext.h
#define HIBAG_STRUCTURE_HEAD_ONLY
#include <LibHLA_ext.h>

#include <string.h>
#include <cstdlib>
#include <cmath>
#include <vector>

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


	// OpenCL memory objects

	// the max number of samples can be hold in mem_prob_buffer
	static int mem_sample_nmax = 0;
	// the max number of haplotypes can be hold in mem_haplo_list
	static int build_haplo_nmax = 0;

	// sizeof(double[nHLA*(nHLA+1)/2]), or sizeof(float[nHLA*(nHLA+1)/2])
	static size_t msize_prob_buffer_each = 0;
	// build:   msize_prob_buffer_total = msize_prob_buffer_each * num_sample
	// predict: msize_prob_buffer_total = msize_prob_buffer_each * num_classifier
	static size_t msize_prob_buffer_total = 0;


	static int wdim_pred_addprob = 0;



	// ===================================================================== //
	// building classifiers

	static int build_num_oob;   ///< the number of out-of-bag samples
	static int build_num_ib;    ///< the number of in-bag samples
	static int run_num_haplo;   ///< the total number of haplotypes
	static int run_num_snp;     ///< the number of SNPs
	static int wdim_num_haplo;  ///< global_work_size for the number of haplotypes


	// ===================================================================== //

	// OpenCL error message
	static const char *err_text(const char *txt, int err)
	{
		static char buf[1024];
		sprintf(buf, "%s (error: %d, %s).", txt, err, gpu_error_info(err));
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
					fn, (long long)(size*sizeof(TYPE)), err, gpu_error_info(err));
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
	cl_int err = clEnqueueNDRangeKernel(gpu_command_queue, gpu_kl_clear_mem, 1, NULL,
		&wdim, &gpu_local_size_d1, 0, NULL, event);
	if (err != CL_SUCCESS)
	{
		Rf_error(
			"Failed to run clEnqueueNDRangeKernel() on 'gpu_kl_clear_mem' (error: %d, %s)",
			err, gpu_error_info(err));
	}
	if (!event) gpu_finish();
#endif
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

// initialize the internal structure for building a model
SEXP ocl_build_init(SEXP R_nHLA, SEXP R_nSample, SEXP R_verbose)
{
	const int n_hla  = Rf_asInteger(R_nHLA);
	const int n_samp = Rf_asInteger(R_nSample);
	ocl_verbose = (Rf_asLogical(R_verbose) == TRUE);
	if (n_hla >= 32768)
		Rf_error("There are too many unique HLA alleles (%d).", n_hla);

	// initialize
	Num_HLA = n_hla;
	Num_Sample = n_samp;
	const int sz_hla = n_hla*(n_hla+1)/2;

	// GPU memory
	const size_t float_size = gpu_f64_build_flag ? sizeof(double) : sizeof(float);
	cl_mem mem_rare_freq = gpu_f64_build_flag ? mem_rare_freq_f64 : mem_rare_freq_f32;

	// allocate
	GPU_CREATE_MEM_V(mem_build_haplo_idx, CL_MEM_READ_WRITE, sizeof(int)*(n_hla+1), NULL);
	GPU_CREATE_MEM_V(mem_build_idx_oob, CL_MEM_READ_WRITE, sizeof(int)*n_samp, NULL);
	GPU_CREATE_MEM_V(mem_build_idx_ib,  CL_MEM_READ_WRITE, sizeof(int)*n_samp, NULL);
	GPU_CREATE_MEM_V(mem_build_output, CL_MEM_READ_WRITE, float_size*n_samp, NULL);
	GPU_CREATE_MEM_V(mem_snpgeno, CL_MEM_READ_WRITE, sizeof(TGenotype)*n_samp, NULL);
	GPU_CREATE_MEM_V(mem_build_hla_idx_map, CL_MEM_READ_WRITE, sizeof(int)*sz_hla, NULL);
	{
		// initialize mem_build_hla_idx_map
		GPU_MEM_MAP(M, int, mem_build_hla_idx_map, sz_hla, false);
		int *p = M.ptr();
		for (int h1=0; h1 < n_hla; h1++)
		{
			for (int h2=h1; h2 < n_hla; h2++)
				*p++ = h1 | (h2 << 16);
		}
	}

	// determine max # of haplo
	if (n_samp <= 250)
		build_haplo_nmax = n_samp * 10;
	else if (n_samp <= 1000L)
		build_haplo_nmax = n_samp * 5;
	else if (n_samp <= 5000)
		build_haplo_nmax = n_samp * 3;
	else if (n_samp <= 10000)
		build_haplo_nmax = (int)(n_samp * 1.5);
	else
		build_haplo_nmax = n_samp;
	GPU_CREATE_MEM_V(mem_haplo_list, CL_MEM_READ_WRITE,
		sizeof(THaplotype)*build_haplo_nmax, NULL);

	// max. # of samples
	msize_prob_buffer_each = sz_hla * float_size;
	mem_sample_nmax = n_samp;
	GPU_CREATE_MEM_V(mem_prob_buffer, CL_MEM_READ_WRITE,
		msize_prob_buffer_each*mem_sample_nmax, NULL);
	msize_prob_buffer_total = msize_prob_buffer_each * mem_sample_nmax;
	cl_uint nmax_buf = msize_prob_buffer_total / sizeof(cl_uint);
	const int zero = 0;


	// arguments for build_haplo_match1
	GPU_SETARG(gpu_kl_build_haplo_match1, 0, mem_build_output);
	GPU_SETARG(gpu_kl_build_haplo_match1, 1, mem_prob_buffer);
	GPU_SETARG(gpu_kl_build_haplo_match1, 2, zero);
	GPU_SETARG_LOCAL(gpu_kl_build_haplo_match1, 3, gpu_local_size_d1*sizeof(cl_uint));
	GPU_SETARG(gpu_kl_build_haplo_match1, 4, nmax_buf);
	GPU_SETARG(gpu_kl_build_haplo_match1, 5, mem_build_idx_ib);
	GPU_SETARG(gpu_kl_build_haplo_match1, 6, mem_build_haplo_idx);
	GPU_SETARG(gpu_kl_build_haplo_match1, 7, mem_haplo_list);
	GPU_SETARG(gpu_kl_build_haplo_match1, 8, mem_snpgeno);

	// arguments for build_haplo_match2
	GPU_SETARG(gpu_kl_build_haplo_match2, 0, mem_build_output);
	GPU_SETARG(gpu_kl_build_haplo_match2, 1, mem_prob_buffer);
	GPU_SETARG(gpu_kl_build_haplo_match2, 2, zero);
	GPU_SETARG_LOCAL(gpu_kl_build_haplo_match2, 3, gpu_local_size_d1*sizeof(cl_uint));
	GPU_SETARG(gpu_kl_build_haplo_match2, 4, nmax_buf);
	GPU_SETARG(gpu_kl_build_haplo_match2, 5, mem_build_idx_ib);
	GPU_SETARG(gpu_kl_build_haplo_match2, 6, mem_build_haplo_idx);
	GPU_SETARG(gpu_kl_build_haplo_match2, 7, mem_haplo_list);
	GPU_SETARG(gpu_kl_build_haplo_match2, 8, mem_snpgeno);

	// arguments for build_calc_prob
	GPU_SETARG(gpu_kl_build_calc_prob, 0, mem_prob_buffer);
	GPU_SETARG(gpu_kl_build_calc_prob, 1, mem_rare_freq);
	GPU_SETARG(gpu_kl_build_calc_prob, 2, n_hla);
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
	GPU_SETARG_LOCAL(gpu_kl_build_calc_oob, 3, gpu_local_size_d1*float_size);
	GPU_SETARG_LOCAL(gpu_kl_build_calc_oob, 4, gpu_local_size_d1*sizeof(int));
	GPU_SETARG(gpu_kl_build_calc_oob, 5, mem_prob_buffer);
	GPU_SETARG(gpu_kl_build_calc_oob, 6, mem_build_hla_idx_map);
	GPU_SETARG(gpu_kl_build_calc_oob, 7, mem_build_idx_oob);
	GPU_SETARG(gpu_kl_build_calc_oob, 8, mem_snpgeno);

	// arguments for gpu_kl_build_calc_ib (in-bag)
	GPU_SETARG(gpu_kl_build_calc_ib, 0, mem_build_output);
	GPU_SETARG(gpu_kl_build_calc_ib, 1, zero);  // start_sample_idx
	if (gpu_f64_build_flag)
	{
		double zero = 0;
		GPU_SETARG(gpu_kl_build_calc_ib, 2, zero);   // aux_log_freq
	} else {
		float zero = 0;
		GPU_SETARG(gpu_kl_build_calc_ib, 2, zero);  // aux_log_freq
	}
	GPU_SETARG(gpu_kl_build_calc_ib, 3, n_hla);
	GPU_SETARG(gpu_kl_build_calc_ib, 4, sz_hla);
	GPU_SETARG_LOCAL(gpu_kl_build_calc_ib, 5, gpu_local_size_d1*float_size);
	GPU_SETARG(gpu_kl_build_calc_ib, 6, mem_prob_buffer);
	GPU_SETARG(gpu_kl_build_calc_ib, 7, mem_build_idx_ib);
	GPU_SETARG(gpu_kl_build_calc_ib, 8, mem_snpgeno);

	// arguments for gpu_kl_build_clear_mem
	GPU_SETARG(gpu_kl_clear_mem, 1, mem_prob_buffer);

	return R_NilValue;
}

SEXP ocl_build_done()
{
	GPU_FREE_MEM(mem_prob_buffer);       mem_prob_buffer = NULL;
	GPU_FREE_MEM(mem_haplo_list);        mem_haplo_list = NULL;
	GPU_FREE_MEM(mem_build_hla_idx_map); mem_build_hla_idx_map = NULL;
	GPU_FREE_MEM(mem_snpgeno);           mem_snpgeno = NULL;
	GPU_FREE_MEM(mem_build_output);      mem_build_output = NULL;
	GPU_FREE_MEM(mem_build_idx_ib);      mem_build_idx_ib = NULL;
	GPU_FREE_MEM(mem_build_idx_oob);     mem_build_idx_oob = NULL;
	GPU_FREE_MEM(mem_build_haplo_idx);   mem_build_haplo_idx = NULL;
	return R_NilValue;
}

// ========

static void build_init(int nHLA, int nSample)
{
	if (nHLA != Num_HLA || Num_Sample != nSample)
		throw "Internal error in build_init()";
}

static void build_done() { }

static void build_set_bootstrap(const int oob_cnt[])
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


static UINT32 *build_haplomatch(const THaplotype haplo[], const int HaploStartIdx[],
	int n_snp, const TGenotype geno[], size_t &out_n)
{
	// find max and check
	int nhla_each_max = 0;
	for (int i=0; i < Num_HLA; i++)
	{
		int m = HaploStartIdx[i+1] - HaploStartIdx[i];
		if (m > nhla_each_max)
			nhla_each_max = m;
	}
	size_t wdim_n_haplo = nhla_each_max;
	if (wdim_n_haplo % gpu_local_size_d2)
		wdim_n_haplo = (wdim_n_haplo/gpu_local_size_d2 + 1)*gpu_local_size_d2;

	const int n_haplo = HaploStartIdx[Num_HLA];
	if (n_haplo > build_haplo_nmax)
		throw "Too many haplotypes out of the limit in build_haplomatch().";

	cl_event events[5];
	events[0] = GPU_WRITE_EVENT(mem_haplo_list, sizeof(THaplotype)*n_haplo, haplo);
	events[1] = GPU_WRITE_EVENT(mem_snpgeno, sizeof(TGenotype)*Num_Sample, geno);
	GPU_SETARG(gpu_kl_clear_mem, 1, mem_build_output); // store MinDiff
	clear_prob_buffer(build_num_ib*sizeof(int), &events[2]);
	events[3] = GPU_WRITE_EVENT(mem_build_haplo_idx, sizeof(int)*(Num_HLA+1), HaploStartIdx);
	int zero = 0;
	events[4] = GPU_WRITE_EVENT(mem_prob_buffer, sizeof(int), &zero);

	// haplotype matching, first pass
	size_t wdims[3] = { (size_t)build_num_ib, wdim_n_haplo, wdim_n_haplo };
	size_t local_size[3] = { 1, gpu_local_size_d2, gpu_local_size_d2 };
	GPU_RUN_KERNEL_EVENT(gpu_kl_build_haplo_match1, 3, wdims, local_size,
		5, events, &events[0]);
	gpu_finish();
	gpu_free_events(5, events);

	// haplotype matching, second pass if MinDiff!=0
	GPU_RUN_KERNEL(gpu_kl_build_haplo_match2, 3, wdims, local_size);

	// output
	UINT32 nbuf=0;
	GPU_READ_MEM(mem_prob_buffer, 0, sizeof(nbuf), &nbuf);
	const size_t sz = sizeof(UINT32)*size_t(nbuf);
	if (sz > msize_prob_buffer_total-sizeof(cl_uint))
		throw "Insuffient GPU buffer in build_haplomatch().";
	UINT32 *buf = (UINT32*)malloc(sz);
	if (!buf)
		throw "Insuffient memory in build_haplomatch().";
	GPU_READ_MEM(mem_prob_buffer, sizeof(UINT32), sz, buf);
	return buf;
}


static void build_set_haplo_geno(const THaplotype haplo[], int n_haplo,
	const TGenotype geno[], int n_snp)
{
	if (n_haplo > build_haplo_nmax)
		throw "Too many haplotypes out of the limit, please contact the package author.";

	cl_event events[2];
	events[0] = GPU_WRITE_EVENT(mem_haplo_list, sizeof(THaplotype)*n_haplo, haplo);
	events[1] = GPU_WRITE_EVENT(mem_snpgeno, sizeof(TGenotype)*Num_Sample, geno);

	run_num_snp = n_snp;
	run_num_haplo = wdim_num_haplo = n_haplo;
	if (wdim_num_haplo % gpu_local_size_d2)
		wdim_num_haplo = (wdim_num_haplo/gpu_local_size_d2 + 1)*gpu_local_size_d2;

	double sum = 0;
	for (int i=0; i < n_haplo; i++) sum += haplo[i].Freq;
	sum = sum / n_haplo;
	double run_aux_log_freq2 = log(sum * sum);
	if (gpu_f64_build_flag)
	{
		GPU_SETARG(gpu_kl_build_calc_ib, 2, run_aux_log_freq2);
	} else {
		float run_aux_log_freq2_f32 = run_aux_log_freq2;
		GPU_SETARG(gpu_kl_build_calc_ib, 2, run_aux_log_freq2_f32);
	}

	gpu_finish();
	gpu_free_events(2, events);
}


static int build_acc_oob()
{
	if (build_num_oob <= 0) return 0;
	if (build_num_oob > mem_sample_nmax)
		throw "Too many sample out of the limit of GPU memory, please contact the package author.";

	// initialize
	cl_event events[4];
	GPU_SETARG(gpu_kl_clear_mem, 1, mem_prob_buffer);
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
		events[2] = GPU_WRITE_EVENT(mem_build_output, sizeof(zero), &zero);

		GPU_SETARG(gpu_kl_build_calc_oob, 1, zero);  // start_sample_idx
		size_t wdims[2] = { gpu_local_size_d1, (size_t)build_num_oob };
		size_t local_size[2] = { gpu_local_size_d1, 1 };
		GPU_RUN_KERNEL_EVENT(gpu_kl_build_calc_oob, 2, wdims, local_size,
			2, &events[1], &events[3]);
	}

	// host waits for GPU
	gpu_finish();
	gpu_free_events(4, events);

	// read output
	int err_cnt = 0;
	GPU_READ_MEM(mem_build_output, 0, sizeof(int), &err_cnt);
	return build_num_oob*2 - err_cnt;
}


static double build_acc_ib()
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
		size_t wdims[2] = { gpu_local_size_d1, (size_t)build_num_ib };
		size_t local_size[2] = { gpu_local_size_d1, 1 };
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
static void predict_init(int n_hla, int nClassifier, const THaplotype *const pHaplo[],
	const int nHaplo[], const int nSNP[])
{
	// assign
	Num_HLA = n_hla;
	Num_Classifier = nClassifier;
	const size_t size_hla = n_hla * (n_hla+1) >> 1;

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
	cl_mem mem_rare_freq = gpu_f64_pred_flag ? mem_rare_freq_f64 : mem_rare_freq_f32;

	// memory for SNP genotypes
	GPU_CREATE_MEM_V(mem_snpgeno, CL_MEM_READ_ONLY, sizeof(TGenotype)*nClassifier, NULL);

	// haplotype lists for all classifiers
	vector<int> nhaplo_buf(4*nClassifier);
	const size_t msize_haplo = sizeof(THaplotype)*sum_n_haplo;
	GPU_CREATE_MEM_V(mem_haplo_list, CL_MEM_READ_ONLY, msize_haplo, NULL);
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
	GPU_CREATE_MEM_V(mem_pred_haplo_num, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(int)*nhaplo_buf.size(), (void*)&nhaplo_buf[0]);

	// pred_calc_prob -- out_prob
	msize_prob_buffer_each = size_hla *
		(gpu_f64_pred_flag ? sizeof(double) : sizeof(float));
	msize_prob_buffer_total = msize_prob_buffer_each * nClassifier;
	GPU_CREATE_MEM_V(mem_prob_buffer, CL_MEM_READ_WRITE, msize_prob_buffer_total, NULL);

	// pred_calc_addprob -- weight
	GPU_CREATE_MEM_V(mem_pred_weight, CL_MEM_READ_WRITE, sizeof(double)*nClassifier,
		NULL);

	// arguments for gpu_kl_pred_calc, pred_calc_prob
	int sz_hla = size_hla;
	GPU_SETARG(gpu_kl_pred_calc, 0, mem_prob_buffer);
	GPU_SETARG(gpu_kl_pred_calc, 1, n_hla);
	GPU_SETARG(gpu_kl_pred_calc, 2, sz_hla);
	GPU_SETARG(gpu_kl_pred_calc, 3, mem_rare_freq);
	GPU_SETARG(gpu_kl_pred_calc, 4, mem_haplo_list);
	GPU_SETARG(gpu_kl_pred_calc, 5, mem_pred_haplo_num);
	GPU_SETARG(gpu_kl_pred_calc, 6, mem_snpgeno);

	// arguments for gpu_kl_pred_sumprob, pred_calc_sumprob
	GPU_SETARG(gpu_kl_pred_sumprob, 0, mem_pred_weight);
	GPU_SETARG(gpu_kl_pred_sumprob, 1, sz_hla);
	GPU_SETARG(gpu_kl_pred_sumprob, 2, mem_prob_buffer);
	GPU_SETARG_LOCAL(gpu_kl_pred_sumprob, 3, sizeof(double)*gpu_local_size_d1);

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
	GPU_FREE_MEM(mem_haplo_list);     mem_haplo_list = NULL;
	GPU_FREE_MEM(mem_pred_haplo_num); mem_pred_haplo_num = NULL;
	GPU_FREE_MEM(mem_snpgeno);        mem_snpgeno = NULL;
	GPU_FREE_MEM(mem_prob_buffer);    mem_prob_buffer = NULL;
	GPU_FREE_MEM(mem_pred_weight);    mem_pred_weight = NULL;
}


/// average the posterior probabilities among classifiers for predicting
void predict_avg_prob(const TGenotype geno[], const double weight[],
	double out_prob[], double out_match[])
{
	const size_t num_size = Num_HLA * (Num_HLA + 1) >> 1;
	cl_event events[4];

	// initialize
	clear_prob_buffer(msize_prob_buffer_total, &events[0]);
	events[1] = GPU_WRITE_EVENT(mem_snpgeno, sizeof(TGenotype)*Num_Classifier, geno);

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
			size_t wdims[2] = { gpu_local_size_d1, (size_t)Num_Classifier };
			size_t local_size[2] = { gpu_local_size_d1, 1 };
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
	memset(&GPU_Proc, 0, sizeof(GPU_Proc));
	GPU_Proc.build_init = build_init;
	GPU_Proc.build_done = build_done;
	GPU_Proc.build_set_bootstrap = build_set_bootstrap;
	GPU_Proc.build_haplomatch = build_haplomatch;
	// GPU_Proc.build_set_haplo_geno = build_set_haplo_geno;
	// GPU_Proc.build_acc_oob = build_acc_oob;
	// GPU_Proc.build_acc_ib = build_acc_ib;
	GPU_Proc.predict_init = predict_init;
	GPU_Proc.predict_done = predict_done;
	GPU_Proc.predict_avg_prob = predict_avg_prob;

	// return the pointer object
	return R_MakeExternalPtr(&GPU_Proc, R_NilValue, R_NilValue);
}

} // extern "C"
