#######################################################################
#
# Package Name: HIBAG.gpu
# Description:
#	HIBAG.gpu -- GPU-based implementation for the HIBAG package
#
# HIBAG R package, HLA Genotype Imputation with Attribute Bagging
# Copyright (C) 2020-2021    Xiuwen Zheng (zhengx@u.washington.edu)
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

code_macro <- "
#define SIZEOF_THAPLO_SHIFT    5
#define SIZEOF_TGENOTYPE       48

#define OFFSET_ALLELE_INDEX    28
#define OFFSET_GENO_BOOTSTRAP  32
#define OFFSET_GENO_HLA_A1     36
#define OFFSET_GENO_HLA_A2     40

#define OFFSET_NUM_HAPLO       0
#define OFFSET_NUM_SNP         1
#define OFFSET_START_SAMP_IDX  2
#define OFFSET_PARAM           3
#define OFFSET_OOB_ACC         4

#define LOCAL_IWORK_MAX        64
"


code_hamm_dist_max <- c(
	`double` = "#define HAMM_DIST_MAX    64",
	`single` = "#define HAMM_DIST_MAX    9",
	`mixed`  = "#define HAMM_DIST_MAX    5",
	`half`   = "#define HAMM_DIST_MAX    2")


code_atomic_add_f32 <- "
#define OFFSET_HAPLO_FREQ    24

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
#define OFFSET_HAPLO_FREQ    16

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
// #pragma OPENCL EXTENSION cl_khr_global_int64_base_atomics : enable
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
#define OFFSET_SECOND_HAPLO    16

inline static int hamming_dist(int n, __global const unsigned char *g,
	__global const unsigned char *h_1, __global const unsigned char *h_2)
{
	__global const uint *h1 = (__global const uint *)h_1;
	__global const uint *h2 = (__global const uint *)h_2;
	__global const uint *s1 = (__global const uint *)g;
	__global const uint *s2 = (__global const uint *)(g + OFFSET_SECOND_HAPLO);
	int ans = 0;

	// for-loop
	for (; n > 0; n-=32)
	{
		uint H1 = *h1++, H2 = *h2++;  // two haplotypes
		uint S1 = *s1++, S2 = *s2++;  // genotypes
		uint M  = S1 | ~S2;           // missing value, 0 is missing
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
	__global numeric *out_prob,
	__constant numeric *exp_log_min_rare_freq,
	const int n_hla, const int num_hla_geno,
	const int n_haplo, const int n_snp, const int start_sample_idx,
	__global const int *p_idx,
	__global const unsigned char *p_haplo,
	__global const unsigned char *p_geno)
{
	const int i1 = get_global_id(1);  // first haplotype index
	const int i2 = get_global_id(2);  // second haplotype index
	if ((i2 < i1) || (i2 >= n_haplo)) return;
	const int ii = get_global_id(0);  // individual index

	// the first haplotype
	__global const unsigned char *p1 = p_haplo + (i1 << SIZEOF_THAPLO_SHIFT);
	// the second haplotype
	__global const unsigned char *p2 = p_haplo + (i2 << SIZEOF_THAPLO_SHIFT);
	// SNP genotype
	__global const unsigned char *pg = p_geno +
		(p_idx[start_sample_idx + ii] * SIZEOF_TGENOTYPE);
	// hamming distance
	int d = hamming_dist(n_snp, pg, p1, p2);
	// since exp_log_min_rare_freq[>HAMM_DIST_MAX] = 0
	if (d <= HAMM_DIST_MAX)
	{
		numeric fq1 = *(__global const numeric*)(p1 + OFFSET_HAPLO_FREQ);
		int h1 = *(__global const int*)(p1 + OFFSET_ALLELE_INDEX);
		numeric fq2 = *(__global const numeric*)(p2 + OFFSET_HAPLO_FREQ);
		int h2 = *(__global const int*)(p2 + OFFSET_ALLELE_INDEX);
		// genotype frequency
		numeric ff = fq1 * fq2;
		if (i1 != i2) ff += ff;
		ff *= exp_log_min_rare_freq[d];  // account for mutation and error rate
		// update
		if (ff > 0)
		{
			int k = h2 + (h1 * ((n_hla << 1) - h1 - 1) >> 1);
			atomic_fadd(&out_prob[num_hla_geno*ii + k], ff);
		}
	}
}
"


code_build_calc_oob <- "
inline static int compare_allele(int P1, int P2, int T1, int T2)
{
	int cnt = 0;
	if ((P1==T1) || (P1==T2))
	{
		cnt = 1;
		if (P1==T1) T1 = -1; else T2 = -1;
	}
	if ((P2==T1) || (P2==T2)) cnt ++;
	return cnt;
}

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
__kernel void build_calc_oob(__global int *out_err_cnt,
	const int start_sample_idx, const int num_hla_geno,
	__global const numeric *prob, __global const int *hla_idx_map,
	__global const int *p_idx,
	__global const unsigned char *p_geno)
{
	__local numeric local_max[LOCAL_IWORK_MAX];
	__local int     local_idx[LOCAL_IWORK_MAX];

	const int localsize = get_local_size(0);  // localsize <= LOCAL_IWORK_MAX
	const int i = get_local_id(0);
	const int i_samp = get_global_id(1);
	prob += num_hla_geno * i_samp;

	numeric max_pb = 0;
	int max_idx = -1;
	for (int k=i; k < num_hla_geno; k+=localsize)
	{
		if (max_pb < prob[k])
			{ max_pb = prob[k]; max_idx = k; }
	}
	if (i < localsize)
	{
		local_max[i] = max_pb;
		local_idx[i] = max_idx;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	// reduced, find max
	for (int n=localsize>>1; n > 0; n >>= 1)
	{
		if (i < n)
		{
			if (local_max[i] < local_max[i+n])
			{
				local_max[i] = local_max[i+n];
				local_idx[i] = local_idx[i+n];
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (i == 0)
	{
		max_idx = local_idx[0];
		if (max_idx >= 0)
		{
			// true alleles
			__global const unsigned char *pg = p_geno +
				(p_idx[start_sample_idx + i_samp] * SIZEOF_TGENOTYPE);
			// aux_hla_type.Allele1, aux_hla_type.Allele2
			__global const int *t = (__global const int *)(pg + OFFSET_GENO_HLA_A1);
			// predicted alleles
			const int ii = hla_idx_map[max_idx];
			const int p1 = ii & 0xFFFF, p2 = ii >> 16;
			// compare
			int cnt = 2 - compare_allele(p1, p2, t[0], t[1]);
			if (cnt > 0)
				atomic_add(out_err_cnt, cnt);  // error count
		} else {
			atomic_add(out_err_cnt, 2);  // error count
		}
	}
}
"


code_build_calc_ib <- "
#ifdef USE_SUM_DOUBLE
#   define TFLOAT    double
#else
#   define TFLOAT    numeric
#endif

__kernel void build_calc_ib(__global numeric *out_prob,
	const int start_sample_idx,
	const int n_hla, const int num_hla_geno,
	__global const numeric *prob, __global const int *p_idx,
	__global const unsigned char *p_geno, const numeric aux_log_freq)
{
	__local TFLOAT local_sum[LOCAL_IWORK_MAX];
	const int localsize = get_local_size(0);  // localsize <= LOCAL_IWORK_MAX

	const int i = get_local_id(0);
	const int i_samp = get_global_id(1);
	prob += num_hla_geno * i_samp;

	TFLOAT sum = 0;
	for (int k=i; k < num_hla_geno; k+=localsize)
		sum += prob[k];
	if (i < localsize) local_sum[i] = sum;
	barrier(CLK_LOCAL_MEM_FENCE);

	// reduced sum of local_sum
	for (int n=localsize>>1; n > 0; n >>= 1)
	{
		if (i < n) local_sum[i] += local_sum[i + n];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (i == 0)
	{
		sum = local_sum[0];
		if (sum > 0)
		{
			// TGenotype
			__global const unsigned char *p = p_geno +
				(p_idx[start_sample_idx + i_samp] * SIZEOF_TGENOTYPE);
			// bootstrap count
			int b = *(__global const int *)(p + OFFSET_GENO_BOOTSTRAP);
			// probability of a HLA genotype
			int h1 = *(__global int *)(p + OFFSET_GENO_HLA_A1);  // aux_hla_type.Allele1
			int h2 = *(__global int *)(p + OFFSET_GENO_HLA_A2);  // aux_hla_type.Allele2
			int k = h2 + (h1 * ((n_hla << 1) - h1 - 1) >> 1);
			// log likelihood
			numeric pb = prob[k];
			if (pb > 0)
			{
				sum = b * log(pb / sum);
			} else {
				sum = b * (-HAMM_DIST_MAX*10 + aux_log_freq - log(sum));
			}
		}
		out_prob[i_samp] = sum;
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
	__global numeric *outProb,
	const int n_hla, const int num_hla_geno,
	__constant numeric *exp_log_min_rare_freq,
	__global const unsigned char *pHaplo,
	__global const int *nHaplo,
	__global const unsigned char *pGeno)
{
	const int ii = get_global_id(0);  // the index of individual classifier
	const int i1 = get_global_id(1);  // first haplotype index
	const int i2 = get_global_id(2);  // second haplotype index
	nHaplo += (ii << 2);
	const int n_haplo = nHaplo[0];    // the number of haplotypes
	if ((i2 < i1) || (i2 >= n_haplo)) return;

	// haplotype list for the classifier ii
	pHaplo += (nHaplo[1] << SIZEOF_THAPLO_SHIFT);
	// the first haplotype
	__global const unsigned char *p1 = pHaplo + (i1 << SIZEOF_THAPLO_SHIFT);
	// the second haplotype
	__global const unsigned char *p2 = pHaplo + (i2 << SIZEOF_THAPLO_SHIFT);
	// hamming distance
	pGeno += ii * SIZEOF_TGENOTYPE;
	const int n_snp = nHaplo[2];    // the number of SNPs in the classifier ii
	int d = hamming_dist(n_snp, pGeno, p1, p2);
	// since exp_log_min_rare_freq[>HAMM_DIST_MAX] = 0
	if (d <= HAMM_DIST_MAX)
	{
		const numeric fq1 = *(__global numeric*)(p1 + OFFSET_HAPLO_FREQ);
		const int h1 = *(__global int*)(p1 + OFFSET_ALLELE_INDEX);
		const numeric fq2 = *(__global numeric*)(p2 + OFFSET_HAPLO_FREQ);
		const int h2 = *(__global int*)(p2 + OFFSET_ALLELE_INDEX);
		// genotype frequency
		numeric ff = fq1 * fq2;
		if (i1 != i2) ff += ff;
		ff *= exp_log_min_rare_freq[d];  // account for mutation and error rate
		// update
		int k = h2 + (h1 * ((n_hla << 1) - h1 - 1) >> 1);
		atomic_fadd(&outProb[num_hla_geno*ii + k], ff);
	}
}
"


code_pred_calc_sumprob <- "
#ifdef USE_SUM_DOUBLE
#   define TFLOAT    double
#else
#   define TFLOAT    numeric
#endif

__kernel void pred_calc_sumprob(__global numeric *out_sum, const int num_hla_geno,
	__global const numeric *prob)
{
	__local TFLOAT local_sum[LOCAL_IWORK_MAX];
	const int localsize = get_local_size(0);

	const int i = get_local_id(0);
	if (i >= localsize) return;

	const int i_cfr = get_global_id(1);
	prob += num_hla_geno * i_cfr;

	TFLOAT sum = 0;
	for (int k=i; k < num_hla_geno; k+=localsize)
		sum += prob[k];
	local_sum[i] = sum;

	barrier(CLK_LOCAL_MEM_FENCE);
	if (i == 0)
	{
		sum = 0;
		for (int i=0; i < localsize; i++) sum += local_sum[i];
		out_sum[i_cfr] = sum;
	}
}
"


code_pred_calc_addprob <- "
#ifdef USE_SUM_DOUBLE
#   define TFLOAT    double
#else
#   define TFLOAT    numeric
#endif

// sum of probabilities among all classifiers
__kernel void pred_calc_addprob(__global numeric *out_prob, const int num_hla_geno,
	const int nClassifier, __global const numeric *weight)
{
	const int i = get_global_id(0);
	if (i < num_hla_geno)
	{
		__global numeric *p = out_prob + i;
		TFLOAT sum = 0;
		for (int j=0; j < nClassifier; j++)
		{
			sum += weight[j] * (*p);
			p += num_hla_geno;
		}
		out_prob[i] = sum;
	}
}
"
