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
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <ctime>
#include <cmath>

#include <Rconfig.h>
#include <R.h>
#include <Rmath.h>


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
}
