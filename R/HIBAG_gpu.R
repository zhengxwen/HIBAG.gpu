#######################################################################
#
# Package Name: HIBAG.gpu
# Description:
#	HIBAG -- GPU-based implementation for the HIBAG package
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.	If not, see <http://www.gnu.org/licenses/>.
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


code_build_model <- "
__kernel void build_calc_prob(
	const int nHLA,
	__constant double *exp_log_min_rare_freq,
	__global int *pParam,
	__global unsigned char *pHaplo,
	__global unsigned char *pGeno,
	__global double *outProb)
{
	const int i1 = get_global_id(0);
	const int i2 = get_global_id(1);
	if (i2 < i1) return;

	const int n_haplo = pParam[0];  // the number of haplotypes
	if (i1 >= n_haplo || i2 >= n_haplo) return;

	// constants
	const size_t sz_haplo = 16;
	const int sz_hla = nHLA * (nHLA + 1) >> 1;
	const int n_snp  = pParam[1];
	const int st_samp = pParam[2];
	const int n_samp  = pParam[3];

	pParam += pParam[4];  // offset pParam

	for (int ii=0; ii < n_samp; ii++)
	{
		// the first haplotype
		__global unsigned char *p1 = pHaplo + (i1 << 5);
		const double fq1 = *(__global double*)(p1 + sz_haplo);
		const int h1 = *(__global int*)(p1 + 28);

		// the second haplotype
		__global unsigned char *p2 = pHaplo + (i2 << 5);
		const double fq2 = *(__global double*)(p2 + sz_haplo);
		const int h2 = *(__global int*)(p2 + 28);

		// SNP genotype
		__global unsigned char *p_geno = pGeno + (pParam[st_samp+ii] << 6);

		// genotype frequency
		int d = hamming_dist(n_snp, p_geno, p1, p2);
		double ff = (i1 != i2) ? (2 * fq1 * fq2) : (fq1 * fq2);
		ff *= exp_log_min_rare_freq[d];  // account for mutation and error rate

		// update
		int k = h2 + (h1 * ((nHLA << 1) - h1 - 1) >> 1);
		atomic_fadd(&outProb[k], ff);

		outProb += sz_hla;
	}
}

// since LibHLA_gpu.cpp: gpu_local_size_d1 = 64
#define LOCAL_SIZE    64

__kernel void build_find_maxprob(const int num_hla_geno,
	const int sz_per_local, __global double *prob, __global int *out_idx)
{
	__local double local_max[LOCAL_SIZE];
	__local int    local_idx[LOCAL_SIZE];

	const int i  = get_local_id(0);
	int j = i * sz_per_local;
	const int i_samp = get_global_id(1);
	prob += num_hla_geno * i_samp;

	double max_pb = 0;
	int max_idx = -1;
	for (size_t n=sz_per_local; n>0 && j<num_hla_geno; n--, j++)
	{
		if (max_pb < prob[j])
			{ max_pb = prob[j]; max_idx = j; }
	}
	if (i < LOCAL_SIZE)
		{ local_max[i] = max_pb; local_idx[i] = max_idx; }

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


__kernel void build_sum_prob(const int nHLA, const int num_hla_geno,
	const int sz_per_local, __global int *pParam, __global unsigned char *pGeno,
	__global double *prob, __global double *out_prob)
{
	__local double local_sum[LOCAL_SIZE];

	const int i  = get_local_id(0);
	int j = i * sz_per_local;
	const int i_samp = get_global_id(1);
	prob += num_hla_geno * i_samp;

	double sum = 0;
	for (size_t n=sz_per_local; n>0 && j<num_hla_geno; n--)
		sum += prob[j++];
	if (i < LOCAL_SIZE) local_sum[i] = sum;

	barrier(CLK_LOCAL_MEM_FENCE);
	if (i == 0)
	{
		sum = 0;
		for (int j=0; j < LOCAL_SIZE; j++)
			sum += local_sum[j];

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


code_predict_prob <- "
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
	const size_t sz_haplo = 16;
	const int sz_hla = nHLA * (nHLA + 1) >> 1;

	for (int i_cfr=0; i_cfr < nClassifier; i_cfr++)
	{
		// the number of haplotypes
		const int n_haplo = nHaplo[0];
		if (i1 < n_haplo && i2 < n_haplo)
		{
			// the first haplotype
			__global unsigned char *p1 = pHaplo + (i1 << 5);
			const double fq1 = *(__global double*)(p1 + sz_haplo);
			const int h1 = *(__global int*)(p1 + 28);

			// the second haplotype
			__global unsigned char *p2 = pHaplo + (i2 << 5);
			const double fq2 = *(__global double*)(p2 + sz_haplo);
			const int h2 = *(__global int*)(p2 + 28);

			// genotype frequency
			int d = hamming_dist(nHaplo[1], pGeno, p1, p2);
			double ff = (i1 != i2) ? (2 * fq1 * fq2) : (fq1 * fq2);
			ff *= exp_log_min_rare_freq[d];  // account for mutation and error rate

			// update
			int k = h2 + (h1 * ((nHLA << 1) - h1 - 1) >> 1);
			atomic_fadd(&outProb[k], ff);
		}
		pHaplo += (n_haplo << 5);
		nHaplo += 2;
		pGeno += 64;
		outProb += sz_hla;
	}
}

// since LibHLA_gpu.cpp: gpu_local_size_d1 = 64
#define LOCAL_SIZE    64

__kernel void pred_calc_sumprob(const int num_hla_geno, const int sz_per_local,
	const int nClassifier, __global double *prob, __global double *out_sum)
{
	__local double local_sum[LOCAL_SIZE];
	const int i  = get_local_id(0);
	int i1 = i * sz_per_local;
	const int i2 = get_global_id(1);

	prob += num_hla_geno * i2;
	double sum = 0;
	for (size_t n=sz_per_local; n>0 && i1<num_hla_geno; n--)
		sum += prob[i1++];
	if (i < LOCAL_SIZE) local_sum[i] = sum;

	barrier(CLK_LOCAL_MEM_FENCE);
	if (i == 0)
	{
		sum = 0;
		for (int i=0; i < LOCAL_SIZE; i++) sum += local_sum[i];
		out_sum[i2] = sum;
	}
}

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

	# GPU platform
	on.exit({ HIBAG:::.hlaClearGPU() })

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

	n.snp <- dim(snp.geno)[1L]	   # Num. of SNPs
	n.samp <- dim(snp.geno)[2L]	   # Num. of samples
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
	mod <- list(n.samp = n.samp, n.snp = n.snp, sample.id = samp.id,
		snp.id = tmp.snp.id, snp.position = tmp.snp.position,
		snp.allele = tmp.snp.allele,
		snp.allele.freq = 0.5*rowMeans(snp.geno, na.rm=TRUE),
		hla.locus = hla$locus, hla.allele = levels(H),
		hla.freq = prop.table(table(H)),
		assembly = as.character(snp$assembly)[1L],
		model = ABmodel,
		appendix = list())
	if (is.na(mod$assembly)) mod$assembly <- "unknown"

	class(mod) <- "hlaAttrBagClass"


    ###################################################################
    # calculate matching statistic
    if (verbose)
        cat("Calculating matching statistic:\n")
    pd <- hlaPredict_gpu(mod, snp.geno, verbose=FALSE)
    mod$matching <- pd$value$matching
    if (verbose)
        print(summary(mod$matching))

    mod
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

	# GPU platform
	on.exit({ HIBAG:::.hlaClearGPU() })

	predict(object, snp, NULL, type, vote, allele.check, match.type,
		same.strand, verbose, proc_ptr=.packageEnv$gpu_proc_ptr)
}



#######################################################################
# GPU utilities
#

hlaGPU_BuildKernel <- function(device, name, code, precision=c("single", "double"))
{
	stopifnot(inherits(device, "clDeviceID"))
	stopifnot(is.character(name))
	stopifnot(is.character(code))
	precision <- match.arg(precision)

	.Call(gpu_build_kernel, device, name, code, precision)
}



hlaGPU_Init <- function(device=NULL, use_double=NA, force=FALSE, verbose=TRUE)
{
	# check
	stopifnot(is.null(device) | inherits(device, "clDeviceID"))
	stopifnot(is.logical(use_double), length(use_double)==1L)
	stopifnot(is.logical(force), length(force)==1L)
	stopifnot(is.logical(verbose), length(verbose)==1L)

	if (is.null(device))
	{
		device <- oclDevices(oclPlatforms()[[1L]])[[1L]]
		if (!inherits(device, "clDeviceID"))
			stop("No available GPU device.")
	}

	exts <- oclInfo(device)$exts

	# support 64-bit floating number or not
	dev_fp64 <- any(grepl("cl_khr_fp64", exts))
	if (dev_fp64)
	{
		if (verbose)
			message("GPU device supports 64-bit floating-point number.")
		if (!grepl("cl_khr_int64_base_atomics", exts))
		{
			if (verbose)
				message("    no support of `cl_khr_int64_base_atomics`")
			if (!isTRUE(force)) dev_fp64 <- FALSE
		}
	} else {
		if (verbose)
			message("GPU device does not support 64-bit floating-point number.")
		use_double <- FALSE
	}

	if (is.na(use_double))
	{
		use_double <- dev_fp64
	} else if (isTRUE(use_double))
	{
		if (!dev_fp64)
		{
			message("Switch to 32-bit floating-point number due to the hardware limit.")
			use_double <- FALSE
		}
	}

	if (verbose)
	{
		if (use_double)
			message("using 64-bit floating-point number")
		else
			message("using 32-bit floating-point number")
	}

	if (use_double)
	{
		.packageEnv$precision <- "double"
		.packageEnv$code_build <- paste(
			"#pragma OPENCL EXTENSION cl_khr_fp64 : enable",
			code_atomic_add_f64,
			code_hamming_dist, code_build_model, collapse="\n")
		.packageEnv$code_predict <- paste(
			"#pragma OPENCL EXTENSION cl_khr_fp64 : enable",
			code_atomic_add_f64,
			code_hamming_dist, code_predict_prob, collapse="\n")
	} else {
		.packageEnv$precision <- "single"
		src <- c("double", "sz_haplo = 16")
		dst <- c("float", "sz_haplo = 24")
		# build
		s <- code_build_model
		for (i in seq_along(src))
			s <- gsub(src[i], dst[i], s, fixed=TRUE)
		.packageEnv$code_build <- paste(
			code_atomic_add_f32, code_hamming_dist, s, collapse="\n")
		# predict
		s <- code_predict_prob
		for (i in seq_along(src))
			s <- gsub(src[i], dst[i], s, fixed=TRUE)
		.packageEnv$code_predict <- paste(
			code_atomic_add_f32, code_hamming_dist, s, collapse="\n")
	}

	# build kernels for constructing classifiers
	k <- hlaGPU_BuildKernel(device,
		c("build_calc_prob", "build_find_maxprob", "build_sum_prob"),
		.packageEnv$code_build, precision=.packageEnv$precision)
	.packageEnv$build_calc_prob <- k[[1L]]
	.packageEnv$build_find_maxprob <- k[[2L]]
	.packageEnv$build_sum_prob <- k[[3L]]

	# build kernels for prediction
	k <- hlaGPU_BuildKernel(device,
		c("pred_calc_prob", "pred_calc_sumprob", "pred_calc_addprob"),
		.packageEnv$code_predict, precision=.packageEnv$precision)
	.packageEnv$kernel_pred <- k[[1L]]
	.packageEnv$kernel_pred_sumprob <- k[[2L]]
	.packageEnv$kernel_pred_addprob <- k[[3L]] 

	# set double floating flag
	.Call(gpu_set_val, 0L, use_double)
	# set OpenCL kernels
	.Call(gpu_set_val,  1L, .packageEnv$build_calc_prob)
	.Call(gpu_set_val,  2L, .packageEnv$build_find_maxprob)
	.Call(gpu_set_val,  3L, .packageEnv$build_sum_prob)
	.Call(gpu_set_val, 11L, .packageEnv$kernel_pred)
	.Call(gpu_set_val, 12L, .packageEnv$kernel_pred_sumprob)
	.Call(gpu_set_val, 13L, .packageEnv$kernel_pred_addprob)

	invisible()
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
	dev_fp64 <- any(grepl("cl_khr_fp64", exts))
	if (dev_fp64)
	{
		packageStartupMessage("GPU device supports 64-bit floating number.")
		if (!grepl("cl_khr_int64_base_atomics", exts))
		{
			packageStartupMessage("But it does not support cl_khr_int64_base_atomics, switch to 32-bit floating number.")
			dev_fp64 <- FALSE
		}
	} else {
		packageStartupMessage("No support of 64-bit floating number and use 32-bit floating number instead.")
	}

	hlaGPU_Init(dev, dev_fp64, verbose=FALSE)

	.packageEnv$gpu_proc_ptr <- .Call(gpu_init_proc)

	TRUE
}
