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
"

code_macro_prec <- c(
	`double` = "#define numeric    double",
	`single` = "#define numeric    float",
	`mixed`  = "#define numeric    float",
	`half`   = "#define numeric    float")

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

#if defined(__OPENCL_VERSION__) && (__OPENCL_VERSION__ >= 120)
#   define POPCNT(v)    v = popcount(v)
#else
#   define POPCNT(v)    \\
		v -= ((v >> 1) & 0x55555555);  \\
		v = (v & 0x33333333) + ((v >> 2) & 0x33333333);  \\
		v = (v + (v >> 4)) & 0x0F0F0F0F;  \\
		v = v + (v >> 8); v = v + (v >> 16);  \\
		v = v & 0x3F
#endif

#define HAMM_CALC(TYPE)    \\
	TYPE S1 = *((__global const TYPE *)geno);    \\
	TYPE S2 = *((__global const TYPE *)(geno + OFFSET_SECOND_HAPLO));  \\
	TYPE MASK = ((H1 ^ S2) | (H2 ^ S1)) & (S1 | ~S2);   \\
	TYPE v1 = (H1 ^ S1) & MASK, v2 = (H2 ^ S2) & MASK;  \\
	POPCNT(v1);  \\
	POPCNT(v2);  \\
	TYPE pn = v1 + v2;


inline static int hamming_dist(__global const unsigned char *geno,
#if defined(FIXED_NUM_INT_HAMM)
#   if FIXED_NUM_INT_HAMM==128
		const uint4 H1, const uint4 H2)
#   elif FIXED_NUM_INT_HAMM==96
		const uint3 H1, const uint3 H2)
#   elif FIXED_NUM_INT_HAMM==64
		const uint2 H1, const uint2 H2)
#   elif FIXED_NUM_INT_HAMM==32
		const uint H1, const uint H2)
#   endif
#else
	int n, __global const unsigned char *h_1, __global const unsigned char *h_2)
#endif
{
#if defined(FIXED_NUM_INT_HAMM) && (FIXED_NUM_INT_HAMM==128)
	HAMM_CALC(uint4)
	return pn.s0 + pn.s1 + pn.s2 + pn.s3;
#elif defined(FIXED_NUM_INT_HAMM) && (FIXED_NUM_INT_HAMM==96)
	HAMM_CALC(uint3)
	return pn.s0 + pn.s1 + pn.s2;
#elif defined(FIXED_NUM_INT_HAMM) && (FIXED_NUM_INT_HAMM==64)
	HAMM_CALC(uint2)
	return pn.s0 + pn.s1;
#elif defined(FIXED_NUM_INT_HAMM) && (FIXED_NUM_INT_HAMM==32)
	HAMM_CALC(uint)
	return pn;
#else
	if (n > 96)  // n always <= 128
	{
		uint4 H1 = *((__global const uint4 *)h_1);
		uint4 H2 = *((__global const uint4 *)h_2);
		HAMM_CALC(uint4)
		return pn.s0 + pn.s1 + pn.s2 + pn.s3;
	} else if (n > 64)
	{
		uint3 H1 = *((__global const uint3 *)h_1);
		uint3 H2 = *((__global const uint3 *)h_2);
		HAMM_CALC(uint3)
		return pn.s0 + pn.s1 + pn.s2;
	} else if (n > 32)
	{
		uint2 H1 = *((__global const uint2 *)h_1);
		uint2 H2 = *((__global const uint2 *)h_2);
		HAMM_CALC(uint2)
		return pn.s0 + pn.s1;
	} else {
		uint H1 = *((__global const uint *)h_1);
		uint H2 = *((__global const uint *)h_2);
		HAMM_CALC(uint)
		return pn;
	}
#endif
}
"


##########################################################################

code_haplo_match_init <- "
__kernel void build_haplo_match_init(const uint n, __global int *p)
{
	const uint i = get_global_id(0);
	if (i < n) p[i] = INT_MAX;
}
"

code_build_alloc_set <- "
inline static void alloc_set(size_t ii, size_t i1, size_t i2,
	__global uint *out_buffer, const uint nmax_buffer)
{
	uint st = atomic_add(out_buffer, 2);  // allocate uint[2]
	if (st < nmax_buffer)
	{
		out_buffer[st] = ii;
		out_buffer[st+1] = i1 | (i2 << 16);
	}
}
"

code_build_haplo_match1 <- "
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
__kernel void build_haplo_match1(
	__global int *out_mindiff, __global uint *out_buffer, const int n_snp,
	const uint nmax_buffer, __global const uint *p_haplo_info,
	__global const int *p_samp_idx, __global const unsigned char *p_haplo,
	__global const unsigned char *p_geno)
{
	const size_t ii = get_global_id(0);  // individual index
	const size_t i1 = get_global_id(1);  // first haplotype
	const size_t i2 = get_global_id(2);  // second haplotype

	uint nn = p_haplo_info[ii];  // bounds
	if (i1 >= (nn & 0xFFFF) || i2 >= (nn >> 16)) return;

	__global const unsigned char *pg = p_geno + (p_samp_idx[ii] * SIZEOF_TGENOTYPE);
	__global const int *h = (__global const int *)(pg + OFFSET_GENO_HLA_A1);
	const size_t h1 = h[0], h2 = h[1];
	const size_t st1 = p_haplo_info[get_global_size(0) + h1];
	if (h1 != h2)
	{
		const size_t st2 = p_haplo_info[get_global_size(0) + h2];
		// a pair of haplotypes
		__global const unsigned char *p1 = p_haplo + ((st1 + i1) << SIZEOF_THAPLO_SHIFT);
		__global const unsigned char *p2 = p_haplo + ((st2 + i2) << SIZEOF_THAPLO_SHIFT);
		// distance
		int d = hamming_dist(pg, n_snp, p1, p2);
		if (d < out_mindiff[ii])
			atomic_min(&out_mindiff[ii], d);
		if (d == 0)
			alloc_set(ii, i1, i2, out_buffer, nmax_buffer);
	} else if (i1 <= i2)
	{
		// a pair of haplotypes
		__global const unsigned char *p1 = p_haplo + ((st1 + i1) << SIZEOF_THAPLO_SHIFT);
		__global const unsigned char *p2 = p_haplo + ((st1 + i2) << SIZEOF_THAPLO_SHIFT);
		// distance
		int d = hamming_dist(pg, n_snp, p1, p2);
		if (d < out_mindiff[ii])
			atomic_min(&out_mindiff[ii], d);
		if (d == 0)
			alloc_set(ii, i1, i2, out_buffer, nmax_buffer);
	}
}
"


code_build_haplo_match2 <- "
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
__kernel void build_haplo_match2(
	__global int *out_mindiff, __global uint *out_buffer, const int n_snp,
	const uint nmax_buffer, __global const uint *p_haplo_info,
	__global const int *p_samp_idx, __global const unsigned char *p_haplo,
	__global const unsigned char *p_geno)
{
	const size_t ii = get_global_id(0);  // individual index
	const int dmin = out_mindiff[ii];
	if (dmin <= 0) return;

	const size_t i1 = get_global_id(1);  // first haplotype
	const size_t i2 = get_global_id(2);  // second haplotype

	uint nn = p_haplo_info[ii];  // bounds
	if (i1 >= (nn & 0xFFFF) || i2 >= (nn >> 16)) return;

	__global const unsigned char *pg = p_geno + (p_samp_idx[ii] * SIZEOF_TGENOTYPE);
	__global const int *h = (__global const int *)(pg + OFFSET_GENO_HLA_A1);
	const size_t h1 = h[0], h2 = h[1];
	const size_t st1 = p_haplo_info[get_global_size(0) + h1];
	if (h1 != h2)
	{
		const size_t st2 = p_haplo_info[get_global_size(0) + h2];
		// a pair of haplotypes
		__global const unsigned char *p1 = p_haplo + ((st1 + i1) << SIZEOF_THAPLO_SHIFT);
		__global const unsigned char *p2 = p_haplo + ((st2 + i2) << SIZEOF_THAPLO_SHIFT);
		// distance
		int d = hamming_dist(pg, n_snp, p1, p2);
		if (d == dmin)
			alloc_set(ii, i1, i2, out_buffer, nmax_buffer);
	} else if (i1 <= i2)
	{
		// a pair of haplotypes
		__global const unsigned char *p1 = p_haplo + ((st1 + i1) << SIZEOF_THAPLO_SHIFT);
		__global const unsigned char *p2 = p_haplo + ((st1 + i2) << SIZEOF_THAPLO_SHIFT);
		// distance
		int d = hamming_dist(pg, n_snp, p1, p2);
		if (d == dmin)
			alloc_set(ii, i1, i2, out_buffer, nmax_buffer);
	}
}
"


code_build_calc_prob <- "
__kernel void build_calc_prob(
	__global numeric *out_prob, __constant numeric *exp_log_min_rare_freq,
	const int n_hla, const int num_hla_geno, const int n_haplo,
	const int start_sample_idx, const int n_samp, __global const int *p_samp_idx,
	__global const unsigned char *p_haplo, __global const unsigned char *p_geno)
{
	const int i1 = get_global_id(0);  // first haplotype index
	const int i2 = get_global_id(1);  // second haplotype index
	if ((i2 < i1) || (i2 >= n_haplo)) return;

	// the first haplotype
	__global const unsigned char *p1 = p_haplo + (i1 << SIZEOF_THAPLO_SHIFT);
	// the second haplotype
	__global const unsigned char *p2 = p_haplo + (i2 << SIZEOF_THAPLO_SHIFT);
	// allele and genotype frequency
	numeric fq1 = *(__global const numeric*)(p1 + OFFSET_HAPLO_FREQ);
	numeric fq2 = *(__global const numeric*)(p2 + OFFSET_HAPLO_FREQ);
	numeric ff = fq1 * fq2;
	if (i1 != i2) ff += ff;
	// allele index (always h1 <= h2)
	int h1 = *(__global const int*)(p1 + OFFSET_ALLELE_INDEX);
	int h2 = *(__global const int*)(p2 + OFFSET_ALLELE_INDEX);
	out_prob += h2 + (h1 * ((n_hla << 1) - h1 - 1) >> 1);
	// HLA-allele-specific haplotypes
#if FIXED_NUM_INT_HAMM==128
	const uint4 H1 = *((__global const uint4 *)p1);
	const uint4 H2 = *((__global const uint4 *)p2);
#elif FIXED_NUM_INT_HAMM==96
	const uint3 H1 = *((__global const uint3 *)p1);
	const uint3 H2 = *((__global const uint3 *)p2);
#elif FIXED_NUM_INT_HAMM==64
	const uint2 H1 = *((__global const uint2 *)p1);
	const uint2 H2 = *((__global const uint2 *)p2);
#elif FIXED_NUM_INT_HAMM==32
	const uint H1 = *((__global const uint *)p1);
	const uint H2 = *((__global const uint *)p2);
#endif

	// for each sample
	for (int ii=0; ii < n_samp; ii++)
	{
		// SNP genotype
		__global const unsigned char *snp_g = p_geno +
			(p_samp_idx[start_sample_idx + ii] * SIZEOF_TGENOTYPE);
		// hamming distance
		int d = hamming_dist(snp_g, H1, H2);
		// since exp_log_min_rare_freq[>HAMM_DIST_MAX] = 0
		if (d <= HAMM_DIST_MAX)
		{
			// account for mutation and error rate
			numeric ff_d = ff * exp_log_min_rare_freq[d];
			if (ff_d > 0)
				atomic_fadd(out_prob, ff_d);
		}
		out_prob += num_hla_geno;
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
	__local numeric *local_max, __local int *local_idx,
	__global const numeric *prob, __global const int *hla_idx_map,
	__global const int *p_idx,
	__global const unsigned char *p_geno)
{
	const int localsize = get_local_size(0);
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
	local_max[i] = max_pb;
	local_idx[i] = max_idx;

	// reduced, find max
	for (int n=localsize>>1; n > 0; n >>= 1)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		if (i < n)
		{
			if (local_max[i] < local_max[i+n])
			{
				local_max[i] = local_max[i+n];
				local_idx[i] = local_idx[i+n];
			}
		}
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
__kernel void build_calc_ib(__global numeric *out_prob,
	const int start_sample_idx, const numeric aux_log_freq,
	const int n_hla, const int num_hla_geno,
	__local numeric *local_sum,
	__global const numeric *prob, __global const int *p_idx,
	__global const unsigned char *p_geno)
{
	const int localsize = get_local_size(0);
	const int i = get_local_id(0);
	const int i_samp = get_global_id(1);
	prob += num_hla_geno * i_samp;

	numeric sum = 0;
	for (int k=i; k < num_hla_geno; k+=localsize)
		sum += prob[k];
	local_sum[i] = sum;

	// reduced sum of local_sum
	for (int n=localsize>>1; n > 0; n >>= 1)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		if (i < n) local_sum[i] += local_sum[i + n];
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
__kernel void clear_memory(const uint n, __global int *p)
{
	const uint i = get_global_id(0);
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
	int d = hamming_dist(pGeno, n_snp, p1, p2);
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
	__global const numeric *prob, __local TFLOAT *local_sum)
{
	const int localsize = get_local_size(0);
	const int i = get_local_id(0);
	const int i_cfr = get_global_id(1);
	prob += num_hla_geno * i_cfr;

	TFLOAT sum = 0;
	for (int k=i; k < num_hla_geno; k+=localsize)
		sum += prob[k];
	local_sum[i] = sum;

	// reduced sum of local_sum
	for (int n=localsize>>1; n > 0; n >>= 1)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		if (i < n) local_sum[i] += local_sum[i + n];
	}

	if (i == 0)
		out_sum[i_cfr] = local_sum[0];
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
