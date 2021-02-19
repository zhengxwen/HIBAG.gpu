#######################################################################
#
# Package Name: HIBAG.gpu
# Description:
#	HIBAG.gpu -- GPU-based implementation for the HIBAG package
#
# HIBAG R package, HLA Genotype Imputation with Attribute Bagging
# Copyright (C) 2021    Xiuwen Zheng (zhengx@u.washington.edu)
# All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.	If not, see <http://www.gnu.org/licenses/>.
#


##########################################################################
#
# OpenCL kernel codes
#

code_atomic_add_f32 <- "
#define SZ_HAPLO    16
#define DIST_MAX    9

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
inline static void atomic_fadd(volatile __global float *addr, float val)
{
	union{
		uint  u32;
		float f32;
	} next, expected, current;
	current.f32 = *addr;
	do{
		expected.f32 = current.f32;
		next.f32     = expected.f32 + val;
		current.u32  = atomic_cmpxchg((volatile __global unsigned int *)addr,
			expected.u32, next.u32);
	} while (current.u32 != expected.u32);
}
"

code_atomic_add_f64 <- "
#define SZ_HAPLO    24
#define DIST_MAX    64

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
inline static void atomic_fadd(volatile __global double *addr, double val)
{
	union{
		ulong  u64;
		double f64;
	} next, expected, current;
	current.f64 = *addr;
	do{
		expected.f64 = current.f64;
		next.f64     = expected.f64 + val;
		current.u64  = atom_cmpxchg((volatile __global ulong*)addr,
			expected.u64, next.u64);
	} while (current.u64 != expected.u64);
}
"


code_hamming_dist <- "
inline static int hamming_dist(int n, __global unsigned char *g,
	__global unsigned char *h_1, __global unsigned char *h_2)
{
	__global uint *h1 = (__global uint *)h_1;
	__global uint *h2 = (__global uint *)h_2;
	__global uint *s1 = (__global uint *)(g + 0);
	__global uint *s2 = (__global uint *)(g + 16);
	__global uint *sM = (__global uint *)(g + 32);
	int ans = 0;

	// for-loop
	for (; n > 0; n-=32)
	{
		uint H1 = *h1++, H2 = *h2++;
		uint S1 = *s1++, S2 = *s2++, M = *sM++;
		uint MASK = ((H1 ^ S2) | (H2 ^ S1)) & M;

		// popcount for '(H1 ^ S1) & MASK'
		uint v1 = (H1 ^ S1) & MASK;
	#if defined(__OPENCL_VERSION__) && (__OPENCL_VERSION__ >= 120)
		ans += popcount(v1);
	#else
		v1 -= ((v1 >> 1) & 0x55555555);
		v1 = (v1 & 0x33333333) + ((v1 >> 2) & 0x33333333);
		ans += (((v1 + (v1 >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
	#endif

		// popcount for '(H2 ^ S2) & MASK'
		uint v2 = (H2 ^ S2) & MASK;
	#if defined(__OPENCL_VERSION__) && (__OPENCL_VERSION__ >= 120)
		ans += popcount(v2);
	#else
		v2 -= ((v2 >> 1) & 0x55555555);
		v2 = (v2 & 0x33333333) + ((v2 >> 2) & 0x33333333);
		ans += (((v2 + (v2 >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
	#endif
	}
	return ans;
}
"


##########################################################################

code_build_calc_prob <- "
__kernel void build_calc_prob(
	__global numeric *outProb,
	const int nHLA,
	__constant numeric *exp_log_min_rare_freq,
	__global int *pParam,
	__global unsigned char *pHaplo,
	__global unsigned char *pGeno)
{
	const int i1 = get_global_id(0);
	const int i2 = get_global_id(1);
	if (i2 < i1) return;

	const int n_haplo = pParam[0];  // the number of haplotypes
	if (i1 >= n_haplo || i2 >= n_haplo) return;

	// constants
	const int sz_hla = nHLA * (nHLA + 1) >> 1;
	const int n_snp  = pParam[1];
	const int st_samp = pParam[2];
	const int n_samp  = pParam[3];

	pParam += pParam[4];  // offset pParam

	for (int ii=0; ii < n_samp; ii++)
	{
		// the first haplotype
		__global unsigned char *p1 = pHaplo + (i1 << 5);
		// the second haplotype
		__global unsigned char *p2 = pHaplo + (i2 << 5);
		// SNP genotype
		__global unsigned char *p_geno = pGeno + (pParam[st_samp+ii] << 6);
		// hamming distance
		int d = hamming_dist(n_snp, p_geno, p1, p2);

		if (d <= DIST_MAX)  // since exp_log_min_rare_freq[>DIST_MAX] = 0
		{
			const double fq1 = *(__global double*)(p1 + SZ_HAPLO);
			const int h1 = *(__global int*)(p1 + 28);
			const double fq2 = *(__global double*)(p2 + SZ_HAPLO);
			const int h2 = *(__global int*)(p2 + 28);
			// genotype frequency
			double ff = (i1 != i2) ? (2 * fq1 * fq2) : (fq1 * fq2);
			ff *= exp_log_min_rare_freq[d];  // account for mutation and error rate
			// update
			int k = h2 + (h1 * ((nHLA << 1) - h1 - 1) >> 1);
			atomic_fadd(&outProb[k], ff);
		}

		outProb += sz_hla;
	}
}
"


code_build_find_maxprob <- "
#define LOCAL_SIZE    64
__kernel void build_find_maxprob(__global int *out_idx, const int num_hla_geno,
	__global const double *prob)
{
	__local double local_max[LOCAL_SIZE];
	__local int    local_idx[LOCAL_SIZE];

	const int i = get_local_id(0);
	const int i_samp = get_global_id(1);
	prob += num_hla_geno * i_samp;

	double max_pb = 0;
	int max_idx = -1;
	for (int k=i; k < num_hla_geno; k+=LOCAL_SIZE)
	{
		if (max_pb < prob[k])
			{ max_pb = prob[k]; max_idx = k; }
	}
	if (i < LOCAL_SIZE)
	{
		local_max[i] = max_pb;
		local_idx[i] = max_idx;
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	if (i == 0)
	{
		max_pb = 0; max_idx = -1;
		for (int j=0; j < LOCAL_SIZE; j++)
		{
			if (max_pb < local_max[j])
			{
				max_pb = local_max[j];
				max_idx = local_idx[j];
			}
		}
		out_idx[i_samp] = max_idx;
	}
}
"


code_build_sum_prob <- "
#define LOCAL_SIZE    64
__kernel void build_sum_prob(const int nHLA, const int num_hla_geno,
	__global int *pParam, __global unsigned char *pGeno, __global double *prob,
	__global double *out_prob)
{
	__local double local_sum[LOCAL_SIZE];

	const int i = get_local_id(0);
	const int i_samp = get_global_id(1);
	prob += num_hla_geno * i_samp;

	double sum = 0;
	for (int k=i; k < num_hla_geno; k+=LOCAL_SIZE)
		sum += prob[k];
	if (i < LOCAL_SIZE) local_sum[i] = sum;

	barrier(CLK_LOCAL_MEM_FENCE);
	if (i == 0)
	{
		sum = 0;
		for (int j=0; j < LOCAL_SIZE; j++) sum += local_sum[j];

		out_prob += (i_samp << 1) + i_samp;
		out_prob[0] = sum;

		// SNP genotype
		pParam += pParam[4];  // offset pParam
		__global unsigned char *p = pGeno + (pParam[i_samp] << 6);

		out_prob[1] = *(__global int *)(p + 48);  // BootstrapCount

		int h1 = *(__global int *)(p + 52);  // aux_hla_type.Allele1
		int h2 = *(__global int *)(p + 56);  // aux_hla_type.Allele2
		int k = h2 + (h1 * ((nHLA << 1) - h1 - 1) >> 1);
		out_prob[2] = prob[k];
	}
}
"


code_clear_memory <- "
// if version < 1.2 or clEnqueueFillBuffer is not available
__kernel void clear_memory(const int n, __global int *p)
{
	const int i = get_global_id(0);
	if (i < n) p[i] = 0;
}
"


##########################################################################

code_pred_calc_prob <- "
__kernel void pred_calc_prob(
	const int nHLA,
	const int nClassifier,
	__constant double *exp_log_min_rare_freq,
	__global unsigned char *pHaplo,
	__global int *nHaplo,
	__global unsigned char *pGeno,
	__global double *outProb)
{
	const int i1 = get_global_id(0);
	const int i2 = get_global_id(1);
	if (i2 < i1) return;

	// constants
	const int sz_hla = nHLA * (nHLA + 1) >> 1;

	for (int i_cfr=0; i_cfr < nClassifier; i_cfr++)
	{
		// the number of haplotypes
		const int n_haplo = nHaplo[0];
		if (i1 < n_haplo && i2 < n_haplo)
		{
			// the first haplotype
			__global unsigned char *p1 = pHaplo + (i1 << 5);
			// the second haplotype
			__global unsigned char *p2 = pHaplo + (i2 << 5);
			// hamming distance
			int d = hamming_dist(nHaplo[1], pGeno, p1, p2);

			if (d <= DIST_MAX)  // since exp_log_min_rare_freq[>DIST_MAX] = 0
			{
				const double fq1 = *(__global double*)(p1 + SZ_HAPLO);
				const int h1 = *(__global int*)(p1 + 28);
				const double fq2 = *(__global double*)(p2 + SZ_HAPLO);
				const int h2 = *(__global int*)(p2 + 28);
				// genotype frequency
				double ff = (i1 != i2) ? (2 * fq1 * fq2) : (fq1 * fq2);
				ff *= exp_log_min_rare_freq[d];  // account for mutation and error rate
				// update
				int k = h2 + (h1 * ((nHLA << 1) - h1 - 1) >> 1);
				atomic_fadd(&outProb[k], ff);
			}
		}
		pHaplo += (n_haplo << 5);
		nHaplo += 2;
		pGeno += 64;
		outProb += sz_hla;
	}
}
"


code_pred_calc_sumprob <- "
// since LibHLA_gpu.cpp: gpu_local_size_d1 = 64
#define LOCAL_SIZE    64

__kernel void pred_calc_sumprob(const int num_hla_geno, __global double *prob,
	__global double *out_sum)
{
	__local double local_sum[LOCAL_SIZE];
	const int i = get_local_id(0);
	const int i_cfr = get_global_id(1);
	prob += num_hla_geno * i_cfr;

	double sum = 0;
	for (int k=i; k < num_hla_geno; k+=LOCAL_SIZE)
		sum += prob[k];
	if (i < LOCAL_SIZE) local_sum[i] = sum;

	barrier(CLK_LOCAL_MEM_FENCE);
	if (i == 0)
	{
		sum = 0;
		for (int i=0; i < LOCAL_SIZE; i++) sum += local_sum[i];
		out_sum[i_cfr] = sum;
	}
}
"


code_pred_calc_addprob <- "
__kernel void pred_calc_addprob(const int num_hla_geno, const int nClassifier,
	__global double *weight, __global double *out_prob)
{
	const int i = get_global_id(0);
	if (i < num_hla_geno)
	{
		__global double *p = out_prob + i;
		double sum = 0;
		for (int j=0; j < nClassifier; j++)
		{
			sum += weight[j] * (*p);
			p += num_hla_geno;
		}
		out_prob[i] = sum;
	}
}
"