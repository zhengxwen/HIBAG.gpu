#######################################################################
#
# Package Name: HIBAG.gpu
# Description:
#	HIBAG.gpu -- GPU-based implementation for the HIBAG package
#
# HIBAG R package, HLA Genotype Imputation with Attribute Bagging
# Copyright (C) 2017-2021    Xiuwen Zheng (zhengx@u.washington.edu)
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

.plural <- function(num) ifelse(num > 1L, "s", "")

.yesno <- function(flag) ifelse(flag, "YES", "NO")


.gpu_create_mem <- function(len, mode, verbose)
{
	sz <- switch(mode, single=4, double=8, integer=4, NA)
	if (verbose) cat("    allocating", sz*len, "bytes in GPU ")
	rv <- clBuffer(.packageEnv$gpu_context, len, mode)
	if (verbose) cat("[OK]\n")
	rv
}

.gpu_build_init_memory <- function(nhla, nsamp, verbose)
{
	# internal
	sizeof_TGenotype  <- 48L
	sizeof_THaplotype <- 32L
	size_hla <- nhla * (nhla+1L) / 2L
	
	# allocate
	.packageEnv$mem_build_idx_oob <- .gpu_create_mem(nsamp, "integer", TRUE)
	.packageEnv$mem_build_idx_ib  <- .gpu_create_mem(nsamp, "integer", TRUE)
	.packageEnv$mem_snpgeno <- .gpu_create_mem(nsamp*sizeof_TGenotype / 4L,
		"integer", TRUE)
	.packageEnv$mem_build_hla_idx_map <- .gpu_create_mem(size_hla,
		"integer", TRUE)
	.packageEnv$mem_build_output <- .gpu_create_mem(nsamp,
		.packageEnv$prec_build, FALSE)

	# determine max # of haplo
	if (nsamp <= 250L)
		build_haplo_nmax <- nsamp * 10L
	else if (nsamp <= 1000L)
		build_haplo_nmax <- nsamp * 5L
	else if (nsamp <= 5000L)
		build_haplo_nmax <- nsamp * 3L
	else if (nsamp <= 10000L)
		build_haplo_nmax <- round(nsamp * 1.5)
	else
		build_haplo_nmax <- nsamp
	.packageEnv$build_haplo_nmax <- build_haplo_nmax
	.packageEnv$mem_haplo_list <- .gpu_create_mem(
		build_haplo_nmax*sizeof_THaplotype / 4L, "integer", verbose)

	build_sample_nmax <- nsamp
	while (build_sample_nmax > 0L)
	{
		ntry <- build_sample_nmax * size_hla
		ok <- tryCatch({
			buffer <- .gpu_create_mem(ntry, .packageEnv$prec_build, verbose)
			TRUE
		}, error=function(e) { cat("[Failed]\n"); FALSE })
		if (ok) break
		build_sample_nmax <- build_sample_nmax - 1000L
		if (build_sample_nmax <= 0L)
			stop(sprintf("Fail to allocate %s[%d] in GPU", .packageEnv$prec_build, ntry))
	}
	.packageEnv$build_sample_nmax <- build_sample_nmax
	.packageEnv$mem_prob_buffer <- buffer

	invisible()
}

.gpu_build_free_memory <- function()
{
	HIBAG:::.hlaClearGPU()
	.Call(gpu_free_memory, .packageEnv$mem_build_idx_oob)
	.Call(gpu_free_memory, .packageEnv$mem_build_idx_ib)
	.Call(gpu_free_memory, .packageEnv$mem_snpgeno)
	.Call(gpu_free_memory, .packageEnv$mem_build_hla_idx_map)
	.Call(gpu_free_memory, .packageEnv$mem_build_output)
	.Call(gpu_free_memory, .packageEnv$mem_haplo_list)
	.Call(gpu_free_memory, .packageEnv$mem_prob_buffer)
	remove(
		mem_build_idx_oob, mem_build_idx_ib, mem_snpgeno, mem_build_hla_idx_map,
		mem_build_output, mem_haplo_list, mem_prob_buffer,
		build_haplo_nmax, build_sample_nmax,
		envir=.packageEnv)
	invisible()
}



##########################################################################
#
# Attribute Bagging method -- HIBAG algorithm
#

hlaAttrBagging_gpu <- function(hla, snp, nclassifier=100,
	mtry=c("sqrt", "all", "one"), prune=TRUE, na.rm=TRUE,
	verbose=TRUE, verbose.detail=FALSE)
{
	# check
	stopifnot(inherits(hla, "hlaAlleleClass"))
	stopifnot(inherits(snp, "hlaSNPGenoClass"))
    stopifnot(is.numeric(nclassifier), length(nclassifier)==1L)
    stopifnot(is.character(mtry) | is.numeric(mtry), length(mtry)>0L)
    stopifnot(is.logical(prune), length(prune)==1L)
    stopifnot(is.logical(na.rm), length(na.rm)==1L)
	stopifnot(is.logical(verbose), length(verbose)==1L)
	stopifnot(is.logical(verbose.detail), length(verbose.detail)==1L)
	if (verbose.detail) verbose <- TRUE

    if (is.na(nclassifier)) nclassifier <- 0L
    with.in.call <- nclassifier==0L
    with.matching <- (nclassifier > 0L)
    if (!with.matching)
    {
        nclassifier <- -nclassifier
        if (nclassifier == 0L) nclassifier <- 1L
    }

	# GPU platform
	on.exit({ .gpu_build_free_memory() })

	# get the common samples
	samp.id <- intersect(hla$value$sample.id, snp$sample.id)

	# hla types
	samp.flag <- match(samp.id, hla$value$sample.id)
	hla.allele1 <- hla$value$allele1[samp.flag]
	hla.allele2 <- hla$value$allele2[samp.flag]
	if (na.rm)
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
        s <- ifelse(!grepl("^KIR", hla$locus), "HLA", "KIR")
        cat("# of unique ", s, " alleles: ", n.hla, "\n", sep="")
		cat("using ", .packageEnv$prec_build,
			"-precision floating-point numbers in GPU computing\n", sep="")
	}


	###################################################################
	# training ...

	.Call(gpu_set_verbose, verbose)
	.Call(gpu_set_local_size)
	.gpu_build_init_memory(n.hla, n.samp, verbose)

	# add new individual classifers
	.Call("HIBAG_NewClassifiers", ABmodel, nclassifier, mtry, prune,
		1L, verbose, verbose.detail, .packageEnv$gpu_proc_ptr, PACKAGE="HIBAG")

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

	# clear memory
	on.exit()
	.gpu_build_free_memory()

	###################################################################
	# calculate matching statistic
    if (with.matching)
    {
		if (verbose)
			cat("Calculating matching proportion:\n")
		pd <- hlaPredict_gpu(mod, snp, verbose=FALSE)
		mod$matching <- pd$value$matching
		if (verbose)
		{
			print(summary(mod$matching))
			acc <- hlaCompareAllele(hla, pd, verbose=FALSE)$overall$acc.haplo
			cat(sprintf("Accuracy with training data: %.1f%%\n", acc*100))
			# out-of-bag accuracy
			mobj <- hlaModelToObj(mod)
			acc <- sapply(mobj$classifiers, function(x) x$outofbag.acc)
			cat(sprintf("Out-of-bag accuracy: %.1f%%\n", mean(acc)*100))
		}
	}

	mod
}



#######################################################################
# Predict HLA types from unphased SNP data
#

hlaPredict_gpu <- function(object, snp,
	type=c("response", "prob", "response+prob"), vote=c("prob", "majority"),
	allele.check=TRUE, match.type=c("Position", "RefSNP+Position", "RefSNP"),
	same.strand=FALSE, verbose=TRUE)
{
	stopifnot(inherits(object, "hlaAttrBagClass"))

	# GPU platform
	on.exit({ HIBAG:::.hlaClearGPU() })
	if (verbose)
	{
		cat("Using ", .packageEnv$prec_predict,
			"-precision floating-point numbers in GPU computing\n", sep="")
	}
	.Call(gpu_set_verbose, verbose)
	.Call(gpu_set_local_size)

	# GPU proc pointer
	cl <- FALSE
	attr(cl, "proc_ptr") <- .packageEnv$gpu_proc_ptr

	# run
	hlaPredict(object, snp, cl, type, vote, allele.check, match.type, same.strand,
		verbose)
}



#######################################################################
# GPU utilities
#

.new_kernel <- function(name, src, mode)
{
	if (mode == "double")
		src <- gsub("numeric", "double", src, fixed=TRUE)
	else if (mode == "single")
		src <- gsub("numeric", "float", src, fixed=TRUE)
	oclSimpleKernel(.packageEnv$gpu_context, name, src, output.mode=mode)
}

# initialize GPU device
.gpu_init <- function(device, use_double, force, dev_idx, showmsg)
{
	# show information
	info <- oclInfo(device)
	if (!is.na(dev_idx) && (dev_idx > 0L))
		s <- paste0("Using Device #", dev_idx, ": ", info$vendor, ", ", info$name)
	else
		s <- paste("Using", info$vendor, info$name)
	showmsg(s)

	# set env variables and need a kernel function
	.packageEnv$gpu_device  <- device
	.packageEnv$gpu_dev_idx <- dev_idx
	.packageEnv$gpu_context <- suppressWarnings(oclContext(device))
	.packageEnv$kernel_clear_mem <- .new_kernel("clear_memory",
		code_clear_memory, "integer")

	if (!is.null(info$driver.ver))
		showmsg(paste("    Driver Version:", info$driver.ver))

	# OpenCL extension
	exts <- info$exts
	test_ext_lst <- c("cl_khr_fp64", "cl_khr_global_int32_base_atomics",
		"cl_khr_int64_base_atomics", "cl_khr_global_int64_base_atomics")
	for (h in test_ext_lst)
		showmsg(paste0("    EXTENSION ", h, ": ", .yesno(any(grepl(h, exts)))))
	has_int64_atom <- any(grepl("cl_khr_int64_base_atomics", exts))

	# support 64-bit floating-point numbers or not
	dev_fp64 <- any(grepl("cl_khr_fp64", exts))
	if (dev_fp64)
	{
		# also need cl_khr_int64_base_atomics : enable
		dev_fp64 <- tryCatch({
			k <- .new_kernel("build_calc_prob", c(code_atomic_add_f64, code_macro,
				code_hamming_dist, code_build_calc_prob), "double")
			TRUE
		}, error=function(e) FALSE)
		showmsg(paste("    atom_cmpxchg (enable cl_khr_int64_base_atomics):",
			ifelse(dev_fp64, "OK", "Failed")))
	}

	# parameters
	pm <- .Call(gpu_get_param)
	showmsg(paste0("    CL_DEVICE_GLOBAL_MEM_SIZE: ",
		prettyNum(pm[[1L]], big.mark=",", scientific=FALSE)))
	showmsg(paste0("    CL_DEVICE_MAX_MEM_ALLOC_SIZE: ",
		prettyNum(pm[[2L]], big.mark=",", scientific=FALSE)))
	showmsg(paste0("    CL_DEVICE_MAX_COMPUTE_UNITS: ", pm[[3L]]))
	showmsg(paste0("    CL_DEVICE_MAX_WORK_GROUP_SIZE: ", pm[[4L]]))
	showmsg(paste0("    CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: ", pm[[5L]]))
	showmsg(paste0("    CL_DEVICE_MAX_WORK_ITEM_SIZES: ", paste(pm[[6L]], collapse=",")))
	showmsg(paste0("    CL_KERNEL_WORK_GROUP_SIZE: ", pm[[7L]]))
	showmsg(paste0("    CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: ", pm[[8L]]))
	showmsg(paste("GPU device", ifelse(dev_fp64, "supports", "does not support"),
		"double-precision floating-point numbers"))

	# user-defined
	if (is.na(use_double) && dev_fp64)
	{
		f64_build <- FALSE
		f64_pred  <- has_int64_atom
		if (f64_pred)
		{
			showmsg(paste("By default, training uses 32-bit floating-point numbers",
				"in GPU and prediction uses 64-bit floating-point numbers",
				"(since EXTENSION cl_khr_int64_base_atomics: YES)."))
		} else {
			showmsg(paste("By default, training and prediction both use 32-bit",
				"floating-point numbers in GPU",
				"(since EXTENSION cl_khr_int64_base_atomics: NO)."))
		}
	} else if (isTRUE(use_double))
	{
		if (!dev_fp64)
		{
			stop("Unable to use 64-bit floating-point numbers in GPU computing.",
				call.=FALSE)
		}
		f64_build <- f64_pred <- TRUE
		showmsg("Training and prediction both use 64-bit floating-point numbers in GPU.")
	} else {
		f64_build <- FALSE
		f64_pred  <- FALSE
		showmsg("Training and prediction both use 32-bit floating-point numbers in GPU.")
	}

	## build OpenCL kernels ##

	.packageEnv$flag_build_f64 <- f64_build
	.packageEnv$flag_pred_f64  <- f64_pred
	.packageEnv$prec_build   <- prec_build   <- ifelse(f64_build, "double", "single")
	.packageEnv$prec_predict <- prec_predict <- ifelse(f64_pred,  "double", "single")

	## build kernels for constructing classifiers
	.packageEnv$kernel_build_calc_prob <- .new_kernel("build_calc_prob",
		c(if (f64_build) code_atomic_add_f64 else code_atomic_add_f32,
			code_macro, code_hamming_dist, code_build_calc_prob), prec_build)
	.packageEnv$kernel_build_calc_oob <- .new_kernel("build_calc_oob",
		c(code_macro, code_build_calc_oob), prec_build)
	.packageEnv$kernel_build_calc_ib <- .new_kernel("build_calc_ib",
		c(code_macro, code_build_calc_ib), prec_build)

	## build kernels for prediction
	.packageEnv$kernel_pred_calc <- .new_kernel("pred_calc_prob",
		c(if (f64_pred) code_atomic_add_f64 else code_atomic_add_f32,
			code_macro, code_hamming_dist, code_pred_calc_prob), prec_predict)
	.packageEnv$kernel_pred_sumprob <- .new_kernel("pred_calc_sumprob",
		c(code_macro, code_pred_calc_sumprob), prec_predict)
	.packageEnv$kernel_pred_addprob <- .new_kernel("pred_calc_addprob",
		code_pred_calc_addprob, prec_predict)

	## initialize GPU memory buffer
	fq <- .Call(gpu_exp_log_min_rare_freq)
	x <- clBuffer(.packageEnv$gpu_context, length(fq), "double")
	x[] <- fq
	.packageEnv$mem_exp_log_min_rare_freq64 <- x
	x <- clBuffer(.packageEnv$gpu_context, length(fq), "single")
	x[] <- fq
	.packageEnv$mem_exp_log_min_rare_freq32 <- x

	invisible()
}


# initialize the internal GPU methods
hlaGPU_Init <- function(device=NA_integer_, use_double=NA, force=FALSE, verbose=TRUE)
{
	# check
	stopifnot(is.numeric(device) | inherits(device, "clDeviceID"))
	stopifnot(is.logical(use_double), length(use_double)==1L)
	stopifnot(is.logical(force), length(force)==1L)
	stopifnot(is.logical(verbose), length(verbose)==1L)

	if (is.na(device))
	{
		device <- .packageEnv$gpu_dev_idx
		if (is.null(device)) device <- NA_integer_
	}
	if (is.na(device))
		device <- .packageEnv$gpu_init_dev_idx

	if (is.numeric(device))
	{
		if (is.null(.packageEnv$opencl_dev_list))
			.get_dev_list(verbose=FALSE)
		num <- device
		if (num < 1L || num > length(.packageEnv$opencl_dev_list))
			stop("No existing device #", num, ".")
		device <- .packageEnv$opencl_dev_list[[num]]
	} else {
		num <- 0L
	}

	.gpu_init(device, use_double, force, num,
		ifelse(verbose, message, function(x) {}))

	invisible()
}



#######################################################################
# Export stardard R library function(s)
#######################################################################

# get a list of OpenCL devices
.get_dev_list <- function(showmsg=message, verbose=TRUE)
{
	.packageEnv$opencl_dev_list <- list()
	# enumate
	i <- 1L
	dev_lst <- list()
	for (pm in oclPlatforms())
	{
		dev <- oclDevices(pm)
		if (length(dev))
		{
			if (verbose)
			{
				for (d in dev)
				{
					m <- oclInfo(d)
					s <- sprintf("Device #%d: %s, %s", i, m$vendor, m$name)
					i <- i + 1L
					showmsg(s)
				}
			}
			dev_lst <- c(dev_lst, dev)
		}
	}
	# set
	.packageEnv$opencl_dev_list <- dev_lst
}


# return device index if found
.search_dev <- function(showmsg=message)
{
	# find NVIDIA, AMD and Intel Graphics cards
	info <- vapply(.packageEnv$opencl_dev_list, function(dev)
		with(oclInfo(dev), paste(vendor, name)), "")
	i1 <- grep("NVIDIA", info, ignore.case=TRUE)
	i2 <- grep("AMD", info, ignore.case=TRUE)
	i3 <- grep("Intel.*Graphics", info, ignore.case=TRUE)
	ii <- c(i1, i2, i3)
	if (length(ii) <= 0L) ii <- 1L
	ii <- c(ii, setdiff(seq_along(info), ii))

	# build OpenCL kernels
	for (i in ii)
	{
		dev <- .packageEnv$opencl_dev_list[[i]]
		ok <- tryCatch({
			# check extension
			exts <- oclInfo(dev)$exts
			if (!any(grepl("cl_khr_global_int32_base_atomics", exts)))
				stop("Need the OpenCL extension cl_khr_global_int32_base_atomics.")
			# initialize
			.packageEnv$gpu_init_dev_idx <- i
			.gpu_init(dev, NA, FALSE, i, showmsg)
			TRUE
		}, error=function(cond) {
			showmsg(cond)
			FALSE
		})
		if (ok) return(i)
	}

	# not found
	NA_integer_
}


.onAttach <- function(libname, pkgname)
{
	# get devices
	packageStartupMessage("")
	packageStartupMessage("Available OpenCL device(s):")
	.get_dev_list(packageStartupMessage, TRUE)

	# find NVIDIA, AMD and Intel Graphics cards
	packageStartupMessage("")
	i <- .search_dev(packageStartupMessage)
	if (is.na(i))
		packageStartupMessage("No device supports!")

	TRUE
}

.onLoad <- function(libname, pkgname)
{
	# set procedure pointer
	.packageEnv$gpu_proc_ptr <- .Call(gpu_init_proc, .packageEnv)
	TRUE
}
