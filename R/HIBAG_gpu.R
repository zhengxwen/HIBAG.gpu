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

code_oob_acc <- "
"

code_predict_prob <- "
	#pragma OPENCL EXTENSION cl_khr_fp64 : enable
	__kernel void name(
		__global double *out_prob,
		const int nHLA,
		const int nClassifier,
		__global unsigned char *pHaplo,
		__global int *nHaplo,
		__global double *tmp_prob)
	{
		int i1 = get_global_id(0);
		int i2 = get_global_id(1);
		if (i2 < i1) return;

		int sz = get_global_size(0);
		int sz_hla = nHLA * (nHLA + 1) >> 1;

		// initialize tmp_prob
		int i = i2 + i1 * (2*sz-i1-1) >> 1;
		if (i < sz_hla) tmp_prob[i] = 0;

		if (i < sz_hla) out_prob[i] = i;
	};
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

	.packageEnv$predict_kernel <- oclSimpleKernel(dev, "name",
		code_predict_prob, precision="double")

	# set double floating flag
	.Call(set_gpu_val, 0L, any(grepl("cl_khr_fp64", oclInfo(dev)$exts)))
	.Call(set_gpu_val, 1L, .packageEnv$predict_kernel)


	.packageEnv$gpu_proc_ptr <- .Call(init_gpu_proc)

	TRUE
}
