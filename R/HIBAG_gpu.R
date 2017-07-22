#######################################################################
#
# Package Name: HIBAG.gpu
# Description:
#   HIBAG -- GPU-based implementation for the HIBAG package
#
# HIBAG R package, HLA Genotype Imputation with Attribute Bagging
# Copyright (C) 2017   Xiuwen Zheng (zhengx@u.washington.edu)
# All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#


##########################################################################

# Package-wide variable
.packageEnv <- new.env()



##########################################################################
#
# OpenCL codes
#

code_atomic_add_f32 <- "
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
		if (n < 32)
			MASK &= ~(((uint)-1) << n);

		// popcount for '(H1 ^ S1) & MASK'
		uint v1 = (H1 ^ S1) & MASK;
		v1 -= ((v1 >> 1) & 0x55555555);
		v1 = (v1 & 0x33333333) + ((v1 >> 2) & 0x33333333);
		ans += (((v1 + (v1 >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;

		// popcount for '(H2 ^ S2) & MASK'
		uint v2 = (H2 ^ S2) & MASK;
		v2 -= ((v2 >> 1) & 0x55555555);
		v2 = (v2 & 0x33333333) + ((v2 >> 2) & 0x33333333);
		ans += (((v2 + (v2 >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
	}
	return ans;
}
"

code_predict_prob <- "
__kernel void pred_calc_prob(
	const int nHLA,
	const int nClassifier,
	__global double *exp_log_min_rare_freq,
	__global unsigned char *pHaplo,
	__global int *nHaplo,
	__global unsigned char *pGeno,
	__global double *out_prob)
{
	const int i1 = get_global_id(0);
	const int i2 = get_global_id(1);
	if (i2 < i1) return;

	// constants
	const size_t sz_haplo = 16;
	const int sz_hla = nHLA * (nHLA + 1) >> 1;

	for (int i_cfr=0; i_cfr < nClassifier; i_cfr++)
	{
		const int n_haplo = nHaplo[i_cfr << 1];  // the number of haplotypes
		if (i1<n_haplo && i2<n_haplo)
		{
			// SNP genotypes
			__global unsigned char *p_geno = pGeno + (i_cfr << 6);
			const int n_snp = nHaplo[(i_cfr << 1) + 1];  // the number of SNPs

			// a pair of haplotypes

			__global unsigned char *p1 = pHaplo + (i1 << 5);
			const double fq1 = *(__global double*)(p1 + sz_haplo);
			const int h1 = *(__global int*)(p1 + 28);

			__global unsigned char *p2 = pHaplo + (i2 << 5);
			const double fq2 = *(__global double*)(p2 + sz_haplo);
			const int h2 = *(__global int*)(p2 + 28);

			int d = hamming_dist(n_snp, p_geno, p1, p2);
			double ff = (i1 != i2) ? (2 * fq1 * fq2) : (fq1 * fq2);
			ff *= exp_log_min_rare_freq[d];

			int k = h2 + (h1 * (2*nHLA - h1 - 1) >> 1);
			atomic_fadd(&out_prob[k], ff);
		}
		// update
		out_prob += sz_hla;
		pHaplo += (n_haplo << 5);
	}
}
"

code_faddmul <- "
__kernel void faddmul_calc(__global unsigned char *buf_param)
{
	// initialize
	const int i = get_global_id(0);
	const int sz = get_global_size(0);

	__global double *p = (__global double *)(buf_param + sizeof(int)*2);
	p += 256;  // since HIBAG_MAXNUM_SNP_IN_CLASSIFIER = 128
	__global double *tmp_prob = p;
	__global double *out_prob = p + sz;
	__global double *p_scale  = out_prob + sz;

	out_prob[i] += tmp_prob[i] * (*p_scale);
}
"




##########################################################################
#
# Attribute Bagging method -- HIBAG algorithm
#

.plural <- function(num)
{
    if (num > 1L) "s" else ""
}

hlaAttrBagging_gpu <- function(hla, snp, nclassifier=100,
    mtry=c("sqrt", "all", "one"), prune=TRUE, rm.na=TRUE,
    verbose=TRUE, verbose.detail=FALSE)
{
    # check
    stopifnot(inherits(hla, "hlaAlleleClass"))
    stopifnot(inherits(snp, "hlaSNPGenoClass"))
    stopifnot(is.character(mtry) | is.numeric(mtry), length(mtry)>0L)
    stopifnot(is.logical(verbose), length(verbose)==1L)
    stopifnot(is.logical(verbose.detail), length(verbose.detail)==1L)
    if (verbose.detail) verbose <- TRUE

    # check GPU platform


    # get the common samples
    samp.id <- intersect(hla$value$sample.id, snp$sample.id)

    # hla types
    samp.flag <- match(samp.id, hla$value$sample.id)
    hla.allele1 <- hla$value$allele1[samp.flag]
    hla.allele2 <- hla$value$allele2[samp.flag]
    if (rm.na)
    {
        if (any(is.na(c(hla.allele1, hla.allele2))))
        {
            warning("There are missing HLA alleles, ",
                "and the corresponding samples have been removed.")
            flag <- is.na(hla.allele1) | is.na(hla.allele2)
            samp.id <- setdiff(samp.id, hla$value$sample.id[samp.flag[flag]])
            samp.flag <- match(samp.id, hla$value$sample.id)
            hla.allele1 <- hla$value$allele1[samp.flag]
            hla.allele2 <- hla$value$allele2[samp.flag]
        }
    } else {
        if (any(is.na(c(hla.allele1, hla.allele2))))
        {
            stop("There are missing HLA alleles!")
        }
    }

    # SNP genotypes
    samp.flag <- match(samp.id, snp$sample.id)
    snp.geno <- snp$genotype[, samp.flag]
    if (!is.integer(snp.geno))
        storage.mode(snp.geno) <- "integer"

    tmp.snp.id <- snp$snp.id
    tmp.snp.position <- snp$snp.position
    tmp.snp.allele <- snp$snp.allele

    # remove mono-SNPs
    snpsel <- rowMeans(snp.geno, na.rm=TRUE)
    snpsel[!is.finite(snpsel)] <- 0
    snpsel <- (0 < snpsel) & (snpsel < 2)
    if (sum(!snpsel) > 0L)
    {
        snp.geno <- snp.geno[snpsel, ]
        if (verbose)
        {
            a <- sum(!snpsel)
            if (a > 0L)
                cat(sprintf("Exclude %d monomorphic SNP%s\n", a, .plural(a)))
        }
        tmp.snp.id <- tmp.snp.id[snpsel]
        tmp.snp.position <- tmp.snp.position[snpsel]
        tmp.snp.allele <- tmp.snp.allele[snpsel]
    }

    if (length(samp.id) <= 0L)
        stop("There is no common sample between 'hla' and 'snp'.")
    if (length(dim(snp.geno)[1L]) <= 0L)
        stop("There is no valid SNP markers.")


    ###################################################################
    # initialize ...

    n.snp <- dim(snp.geno)[1L]     # Num. of SNPs
    n.samp <- dim(snp.geno)[2L]    # Num. of samples
    HUA <- hlaUniqueAllele(c(hla.allele1, hla.allele2))
    H <- factor(match(c(hla.allele1, hla.allele2), HUA))
    levels(H) <- HUA
    n.hla <- nlevels(H)
    H1 <- as.integer(H[1L:n.samp]) - 1L
    H2 <- as.integer(H[(n.samp+1L):(2L*n.samp)]) - 1L

    # create an attribute bagging object (return an integer)
    ABmodel <- .Call("HIBAG_Training", n.snp, n.samp, snp.geno, n.hla, H1, H2,
        PACKAGE="HIBAG")

    # number of variables randomly sampled as candidates at each split
    mtry <- mtry[1L]
    if (is.character(mtry))
    {
        if (mtry == "sqrt")
        {
            mtry <- ceiling(sqrt(n.snp))
        } else if (mtry == "all")
        {
            mtry <- n.snp
        } else if (mtry == "one")
        {
            mtry <- 1L
        } else {
            stop("Invalid mtry!")
        }
    } else if (is.numeric(mtry))
    {
        if (is.finite(mtry))
        {
            if ((0 < mtry) & (mtry < 1)) mtry <- n.snp*mtry
            mtry <- ceiling(mtry)
            if (mtry > n.snp) mtry <- n.snp
        } else {
            mtry <- ceiling(sqrt(n.snp))
        }
    } else {
        stop("Invalid mtry value!")
    }
    if (mtry <= 0) mtry <- 1L

    if (verbose)
    {
        cat(sprintf("Build a HIBAG model with %d individual classifier%s:\n",
            nclassifier, .plural(nclassifier)))
        cat("# of SNPs randomly sampled as candidates for each selection: ",
            mtry, "\n", sep="")
        cat("# of SNPs: ", n.snp, ", # of samples: ", n.samp, "\n", sep="")
        cat("# of unique HLA alleles: ", n.hla, "\n", sep="")
    }


    ###################################################################
    # training ...
    # add new individual classifers
    .Call("HIBAG_NewClassifiers", ABmodel, nclassifier, mtry, prune,
        verbose, verbose.detail, .packageEnv$gpu_proc_ptr, PACKAGE="HIBAG")

    # output
    rv <- list(n.samp = n.samp, n.snp = n.snp, sample.id = samp.id,
        snp.id = tmp.snp.id, snp.position = tmp.snp.position,
        snp.allele = tmp.snp.allele,
        snp.allele.freq = 0.5*rowMeans(snp.geno, na.rm=TRUE),
        hla.locus = hla$locus, hla.allele = levels(H),
        hla.freq = prop.table(table(H)),
        assembly = as.character(snp$assembly)[1L],
        model = ABmodel,
        appendix = list())
    if (is.na(rv$assembly)) rv$assembly <- "unknown"

    class(rv) <- "hlaAttrBagClass"
    rv
}



#######################################################################
# Predict HLA types from unphased SNP data
#

hlaPredict_gpu <- function(object, snp,
    type=c("response", "prob", "response+prob"), vote=c("prob", "majority"),
    allele.check=TRUE, match.type=c("RefSNP+Position", "RefSNP", "Position"),
    same.strand=FALSE, verbose=TRUE)
{
    stopifnot(inherits(object, "hlaAttrBagClass"))
    predict(object, snp, NULL, type, vote, allele.check, match.type,
        same.strand, verbose, proc_ptr=.packageEnv$gpu_proc_ptr)
}



#######################################################################
# Export stardard R library function(s)
#######################################################################

.onAttach <- function(lib, pkg)
{
	platform <- oclPlatforms()
	for (i in seq_along(platform))
	{
		ii <- oclInfo(platform[[i]])
		s <- paste0("Available OpenCL platform: ", ii$name, ", ", ii$version)
		packageStartupMessage(s)
		dev <- oclDevices(platform[[i]])
		for (j in seq_along(dev))
		{
			ii <- oclInfo(dev[[i]])
			s <- paste0("    Device: ", ii$vendor, " ", ii$name)
			packageStartupMessage(s)
		}
	}

	# build OpenCL kernels
	dev <- oclDevices(oclPlatforms()[[1L]])[[1L]]

	# check extension
	exts <- oclInfo(dev)$exts
	if (!grepl("cl_khr_global_int32_base_atomics", exts))
		stop("Need the OpenCL extension cl_khr_global_int32_base_atomics.")

	# support 64-bit floating number or not
	dev_fp64 <- any(grepl("cl_khr_fp64", oclInfo(dev)$exts))
	if (dev_fp64)
	{
		packageStartupMessage("GPU device supports 64-bit floating number.")
		if (!grepl("cl_khr_int64_base_atomics", exts))
		{
			packageStartupMessage("But it does not support cl_khr_int64_base_atomics, switch to 32-bit floating number.")
			dev_fp64 <- FALSE
		}
	} else {
		packageStartupMessage("No support of GPU 64-bit floating number, ",
			"and use GPU 32-bit floating number instead.")
	}

	if (dev_fp64)
	{
		.packageEnv$precision <- "double"
		.packageEnv$code_predict <- paste(
			"#pragma OPENCL EXTENSION cl_khr_fp64 : enable",
			code_atomic_add_f64,
			code_hamming_dist, code_predict_prob, collapse="\n")
	} else {
		.packageEnv$precision <- "single"
		s <- code_predict_prob
		src <- c("double", "sz_haplo = 16")
		dst <- c("float", "sz_haplo = 24")
		for (i in seq_along(src))
			s <- gsub(src[i], dst[i], s, fixed=TRUE)
		.packageEnv$code_predict <- paste(
			code_atomic_add_f32,
			code_hamming_dist, s, collapse="\n")
	}

	.packageEnv$predict_kernel <- oclSimpleKernel(dev, "pred_calc_prob",
		.packageEnv$code_predict, precision=.packageEnv$precision)

	# set double floating flag
	.Call(set_gpu_val, 0L, dev_fp64)
	.Call(set_gpu_val, 1L, .packageEnv$predict_kernel)

	.packageEnv$gpu_proc_ptr <- .Call(init_gpu_proc)

	TRUE
}
