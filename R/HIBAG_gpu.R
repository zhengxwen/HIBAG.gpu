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


.gpu_build_init_memory <- function(nhla, nsamp, verbose)
{
	.Call(ocl_build_init, nhla, nsamp, verbose)
	invisible()
}

.gpu_build_free_memory <- function()
{
	HIBAG:::.hlaClearGPU()
	.Call(ocl_build_done)
	invisible()
}

.set_mtry <- function(mtry, n.snp)
{
	mtry <- mtry[1L]
	if (is.character(mtry))
	{
		if (mtry == "sqrt")
			mtry <- ceiling(sqrt(n.snp))
		else if (mtry == "all")
			mtry <- n.snp
		else if (mtry == "one")
			mtry <- 1L
		else
			stop("Invalid mtry!")
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
	if (mtry <= 0L) mtry <- 1L
	mtry
}


##########################################################################
#
# Attribute Bagging method -- HIBAG algorithm
#

hlaAttrBagging_gpu <- function(hla, snp, nclassifier=100L,
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
	on.exit({ if (!with.in.call) .gpu_build_free_memory() })

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
			stop("There are missing HLA alleles!")
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
	mtry <- .set_mtry(mtry, n.snp)

	if (verbose)
	{
		cat(sprintf("Build a HIBAG model with %d individual classifier%s:\n",
			nclassifier, .plural(nclassifier)))
		cat("# of SNPs randomly sampled as candidates for each selection: ",
			mtry, "\n", sep="")
		cat("# of SNPs: ", n.snp, ", # of samples: ", n.samp, "\n", sep="")
        s <- ifelse(!grepl("^KIR", hla$locus), "HLA", "KIR")
        cat("# of unique ", s, " alleles: ", n.hla, "\n", sep="")
		cat("using ", .packageEnv$train_prec,
			"-precision floating-point numbers in GPU computing\n", sep="")
	}


	###################################################################
	# training ...

	# initialize gpu memory
	if (!with.in.call)
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
	if (!with.in.call)
		.gpu_build_free_memory()

	###################################################################
	# calculate matching statistic
    if (with.matching)
    {
		if (verbose)
			cat("Calculating matching proportion:\n")
		pd <- hlaPredict_gpu(mod, snp, match.type="Pos+Allele", verbose=FALSE)
		mod$matching <- pd$value$matching
		if (verbose)
		{
			print(summary(mod$matching))
			acc <- hlaCompareAllele(hla, pd, verbose=FALSE)$overall$acc.haplo
			cat(sprintf("Accuracy with training data: %.2f%%\n", acc*100))
			# out-of-bag accuracy
			mobj <- hlaModelToObj(mod)
			acc <- sapply(mobj$classifiers, function(x) x$outofbag.acc)
			cat(sprintf("Out-of-bag accuracy: %.2f%%\n", mean(acc)*100))
		}
	}

	mod
}


#######################################################################

postNode <- function(con, type, value=NULL, tag=NULL)
{
	parallel:::sendData(con, list(type=type, data=value, tag=tag))
}
sendCall <- function(con, fun, args, return=TRUE, tag=NULL)
{
	postNode(con, "EXEC", list(fun=fun, args=args, return=return, tag=tag))
	NULL
}
recvOneResult <- function(cl)
{
	v <- parallel:::recvOneData(cl)
	list(value=v$value$value, node=v$node, tag=v$value$tag)
}

.DynamicClusterCall <- function(cl, ntot, fun, update_fc, ...)
{
	# check
	stopifnot(is.null(cl) | inherits(cl, "cluster"))
	stopifnot(is.numeric(ntot), length(ntot)==1L)
	stopifnot(is.function(fun))
	stopifnot(is.function(update_fc))

	p <- length(cl)
	if (ntot > 0L && p > 0L)
	{
		## this closure is sending to all nodes
		argfun <- function(node, i) c(node, i, list(...))
		submit <- function(node, idx)
			sendCall(cl[[node]], fun, argfun(node, idx), tag=idx)

		for (i in 1L:min(ntot, p)) submit(i, i)
		for (i in 1L:ntot)
		{
			d <- recvOneResult(cl)
			j <- i + min(ntot, p)
			if (j <= ntot) submit(d$node, j)
			dv <- d$value
			if (inherits(dv, "try-error"))
				stop("One node produced an error: ", as.character(dv))
			update_fc(d$node, dv)
		}
	} else {
		for (i in seq_len(ntot))
		{
			dv <- fun(1L, i, ...)
			update_fc(i, dv)
		}
	}
	invisible()
}


hlaAttrBagging_MultiGPU <- function(gpus, hla, snp, auto.save="", nclassifier=100L,
	mtry=c("sqrt", "all", "one"), prune=TRUE, na.rm=TRUE,
	train_prec="auto", verbose=TRUE)
{
	# check
	stopifnot(is.numeric(gpus), all(!is.na(gpus)), length(gpus)>0L)
	stopifnot(inherits(hla, "hlaAlleleClass"))
	stopifnot(inherits(snp, "hlaSNPGenoClass"))
	stopifnot(is.character(auto.save), length(auto.save)==1L, !is.na(auto.save))
	if (auto.save!="" && is.na(HIBAG:::.fn_obj_check(auto.save)))
		stop("'auto.save' should be a .rda/.RData or .rds file name.")
	stopifnot(is.numeric(nclassifier), length(nclassifier)==1L, nclassifier>0L)
	stopifnot(is.character(mtry) | is.numeric(mtry), length(mtry)>0L)
	stopifnot(is.logical(prune), length(prune)==1L)
	stopifnot(is.logical(na.rm), length(na.rm)==1L)
	stopifnot(is.logical(verbose), length(verbose)==1L)
	stopifnot(is.character(train_prec),
		length(train_prec)==1L || length(train_prec)==length(gpus))

	# GPU platform
	for (i in gpus)
	{
		n <- .packageEnv$opencl_dev_num
		if (!all(1<=gpus & gpus<=n))
			stop("'gpus' should be between 1 and ", n, ".")
	}
	if (length(train_prec)==1L)
		train_prec <- rep(train_prec, length(gpus))

	if (verbose)
	{
		cat("Building a HIBAG model:\n")
		cat(sprintf("    %d individual classifier%s\n", nclassifier,
			.plural(nclassifier)))
		if (length(gpus) > 1L)
		{
			cat(sprintf("    run in parallel with %d compute node%s\n",
				length(gpus), .plural(length(gpus))))
		}
		if (auto.save != "")
			cat("    autosave to ", sQuote(auto.save), "\n", sep="")
	}

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
			stop("There are missing HLA alleles!")
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

	# initialize ...
	n.snp <- dim(snp.geno)[1L]	   # Num. of SNPs
	n.samp <- dim(snp.geno)[2L]	   # Num. of samples
	HUA <- hlaUniqueAllele(c(hla.allele1, hla.allele2))
	H <- factor(match(c(hla.allele1, hla.allele2), HUA))
	levels(H) <- HUA
	n.hla <- nlevels(H)
	H1 <- as.integer(H[1L:n.samp]) - 1L
	H2 <- as.integer(H[(n.samp+1L):(2L*n.samp)]) - 1L
	mtry <- .set_mtry(mtry, n.snp)
	if (verbose)
	{
		cat("    # of SNPs randomly sampled as candidates for each selection: ",
			mtry, "\n", sep="")
		cat("    # of SNPs: ", n.snp, "\n", sep="")
		cat("    # of samples: ", n.samp, "\n", sep="")
		s <- ifelse(!grepl("^KIR", hla$locus), "HLA", "KIR")
		cat("    # of unique ", s, " alleles: ", n.hla, "\n", sep="")
	}


	# initialize GPUs
	if (length(gpus) == 1L)
	{
		cl <- NULL
		hlaGPU_Init(gpus[1L], train_prec=train_prec, .packageEnv$predict_prec,
			verbose=verbose)

		# create an attribute bagging object (return an integer)
		.packageEnv$M_ABmodel <- .Call("HIBAG_Training", n.snp, n.samp, snp.geno,
			n.hla, H1, H2, PACKAGE="HIBAG")
		.packageEnv$M_mtry  <- mtry
		.packageEnv$M_prune <- prune
		.packageEnv$M_hla.allele <- HUA
		with(.packageEnv, .Call(multigpu_init,
			M_ABmodel, M_mtry, M_prune, M_hla.allele, gpu_proc_ptr))

		.gpu_build_init_memory(n.hla, n.samp, verbose)
		on.exit({
			.packageEnv$Model_ABmodel <- NULL
			.gpu_build_free_memory()
		})

	} else {
		cl <- makeCluster(length(gpus), outfile="", useXDR=FALSE)
		on.exit(stopCluster(cl))
		# GPU
		msg <- clusterApply(cl, seq_along(gpus),
			function(i, gpus, train_prec)
			{
				env <- new.env()
				.gpu_init(gpus[i], train_prec[i], "auto", function(s) env$msg <- s)
				env$msg
			}, gpus=gpus, train_prec=train_prec)
		if (verbose)
		{
			for (i in seq_along(msg))
			{
				ss <- unlist(strsplit(msg[[i]], "\n"))
				ss <- gsub("^    ", ">>  ", ss)
				ss[1L] <- paste0("[[Process ", i, "]]: ", ss[1L])
				ss <- ss[-length(ss)]
				ss[length(ss)] <- paste0("=== ", ss[length(ss)], " ===")
				ss[-1L] <- paste0("    ", ss[-1L])
				cat(paste(ss, collapse="\n"))
				cat("\n")
			}
		}
		# GPU memory
		for (idx in seq_along(gpus))
		{
			clusterApply(cl, seq_along(gpus), function(i, idx, verbose)
				{
					if (i == idx)
					{
						if (verbose)
							cat("[[Process ", i, "]]:\n", sep="")
						.gpu_build_init_memory(n.hla, n.samp, verbose)
					}
				}, idx=idx, verbose=verbose)
		}
		# create an attribute bagging object
		clusterApply(cl, seq_along(gpus),
			function(i, n.snp, n.samp, snp.geno, n.hla, H1, H2, mtry, prune, hla.allele)
			{
				.packageEnv$M_snp.geno <- snp.geno
				.packageEnv$M_H1 <- H1
				.packageEnv$M_H2 <- H2
				.packageEnv$M_mtry  <- mtry
				.packageEnv$M_prune <- prune
				.packageEnv$M_hla.allele <- hla.allele
				.packageEnv$M_ABmodel <- .Call("HIBAG_Training", n.snp, n.samp,
					snp.geno, n.hla, H1, H2, PACKAGE="HIBAG")
				with(.packageEnv, .Call(multigpu_init,
					M_ABmodel, M_mtry, M_prune, M_hla.allele, gpu_proc_ptr))
			}, n.snp=n.snp, n.samp=n.samp, snp.geno=snp.geno, n.hla=n.hla,
				H1=H1, H2=H2, mtry=mtry, prune=prune, hla.allele=HUA)
		# on exit in case fails
		on.exit({
			clusterApply(cl, seq_along(gpus), function(i) .gpu_build_free_memory())
			stopCluster(cl)
		})
		# set random number for the cluster
		RNGkind("L'Ecuyer-CMRG")
		rand <- .Random.seed
		clusterSetRNGStream(cl)
	}


	# output object
	mobj <- list(n.samp = n.samp, n.snp = n.snp, sample.id = samp.id,
		snp.id = tmp.snp.id, snp.position = tmp.snp.position,
		snp.allele = tmp.snp.allele,
		snp.allele.freq = 0.5*rowMeans(snp.geno, na.rm=TRUE),
		hla.locus = hla$locus, hla.allele = levels(H),
		hla.freq = prop.table(table(H)),
		assembly = as.character(snp$assembly)[1L],
        classifiers = list(),
        matching = NULL, appendix = list())
	if (is.na(mobj$assembly)) mobj$assembly <- "unknown"
	class(mobj) <- "hlaAttrBagObj"

	if (verbose)
		cat("[-] ", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n", sep="")
	save_ptm <- proc.time()
	clr <- list()

	.DynamicClusterCall(cl, nclassifier,
		fun = function(node, idx) .Call(multigpu_train, node, idx),
		update_fc = function(job, val)
		{
			if (is.character(val))
			{
				if (verbose) cat(val)
			} else {
				if (verbose) cat(val[[1L]])
				clr <<- append(clr, val[[2L]])
				tm <- proc.time()
				if ((tm - save_ptm)[3L] >= 600L)  # elapsed time >= 10min
				{
					save_ptm <<- tm
					mobj$classifiers <<- clr
					if (auto.save != "")
					{
						HIBAG:::.fn_obj_save(auto.save, mobj)
						HIBAG:::.show_model_obj(mobj, TRUE)
					}
				}
			}
		}
	)

	old_n <- length(clr)
	if (is.null(cl))
	{
		v <- with(.packageEnv, .Call("HIBAG_GetClassifierList",
			M_ABmodel, M_hla.allele, PACKAGE="HIBAG"))
		if (length(v)) clr <- append(clr, v)
		on.exit()
		.packageEnv$M_ABmodel <- NULL
		.gpu_build_free_memory()
	} else {
		on.exit()
		vs <- clusterApply(cl, seq_along(gpus), function(i)
			{
				.gpu_build_free_memory()
				with(.packageEnv, .Call("HIBAG_GetClassifierList",
					M_ABmodel, M_hla.allele, PACKAGE="HIBAG"))
			})
		for (v in vs) clr <- append(clr, v)
		# the next random seed
		nextRNGStream(rand)
		nextRNGSubStream(rand)
		# stop the workers
		stopCluster(cl)
	}

	if (old_n < length(clr))
	{
		mobj$classifiers <- clr
		if (auto.save != "")
		{
			HIBAG:::.fn_obj_save(auto.save, mobj)
			HIBAG:::.show_model_obj(mobj, TRUE)
		}
	}
	mod <- hlaModelFromObj(mobj)


	# matching proportion
	if (verbose)
		cat("Calculating matching proportion:\n")
	pd <- hlaPredict_gpu(mod, snp, match.type="Pos+Allele", verbose=FALSE)
	mod$matching <- pd$value$matching
	mobj <- NULL
	if (auto.save != "")
	{
		mobj <- hlaModelToObj(mod)
		HIBAG:::.fn_obj_save(auto.save, mobj)
	}
	if (verbose)
	{
		HIBAG:::.printMatching(mod$matching)
		acc <- hlaCompareAllele(hla, pd, verbose=FALSE)$overall$acc.haplo
		cat(sprintf("Accuracy with training data: %.2f%%\n", acc*100))
		# out-of-bag accuracy
		if (is.null(mobj)) mobj <- hlaModelToObj(mod)
		acc <- sapply(mobj$classifiers, function(x) x$outofbag.acc)
		cat(sprintf("Out-of-bag accuracy: %.2f%%\n", mean(acc)*100))
	}

	# output
	if (auto.save != "") invisible() else mod
}


#######################################################################
# Predict HLA types from unphased SNP data
#

hlaPredict_gpu <- function(model, snp,
	type=c("response", "dosage", "prob", "response+prob"), vote=c("prob", "majority"),
	allele.check=TRUE, match.type=c("Position", "Pos+Allele", "RefSNP+Position", "RefSNP"),
	same.strand=FALSE, verbose=TRUE)
{
	stopifnot(inherits(model, "hlaAttrBagClass"))
	type <- match.arg(type)
	vote <- match.arg(vote)
	match.type <- match.arg(match.type)

	# GPU platform
	on.exit({ HIBAG:::.hlaClearGPU() })
	if (verbose)
	{
		cat("Using ", .packageEnv$predict_prec,
			"-precision floating-point numbers in GPU computing\n", sep="")
	}
	.Call(ocl_set_verbose, verbose)

	# GPU proc pointer
	cl <- FALSE
	attr(cl, "proc_ptr") <- .packageEnv$gpu_proc_ptr

	# run
	hlaPredict(model, snp, cl=cl, type=type, vote=vote, allele.check=allele.check,
		match.type=match.type, same.strand=same.strand, verbose=verbose)
}



#######################################################################
# GPU utilities
#

# initialize GPU device
.gpu_init <- function(dev_idx, train_prec, predict_prec, showmsg)
{
	# show information
	info <- .get_dev_info(dev_idx)
	msg <- paste0("Using Device #", dev_idx, ": ", info[[1L]], ", ", info[[2L]])

	# set env variables
	.packageEnv$gpu_dev_idx <- dev_idx
	s <- .Call(ocl_select_dev, dev_idx)
	ss <- info[6L]
	if (!grepl("\\s$", ss)) ss <- paste0(ss, " ")
	msg <- c(msg, paste0("    Device Version: ", ss, "(", s, ")"))
	msg <- c(msg, paste("    Driver Version:", info[7L]))
	on.exit(showmsg(paste(msg, collapse="\n")))

	# OpenCL extension
	exts <- info[4L]
	test_ext_lst <- c("cl_khr_global_int32_base_atomics", "cl_khr_fp64",
		"cl_khr_int64_base_atomics", "cl_khr_global_int64_base_atomics")
	for (h in test_ext_lst)
		msg <- c(msg, paste0("    EXTENSION ", h, ": ", .yesno(grepl(h, exts))))
	has_int64_atom <- grepl("cl_khr_int64_base_atomics", exts)

	pm <- .Call(ocl_get_dev_param)
	msg <- c(msg, paste0("    CL_DEVICE_GLOBAL_MEM_SIZE: ",
		prettyNum(pm[[1L]], big.mark=",", scientific=FALSE)))
	msg <- c(msg, paste0("    CL_DEVICE_MAX_MEM_ALLOC_SIZE: ",
		prettyNum(pm[[2L]], big.mark=",", scientific=FALSE)))
	msg <- c(msg, paste0("    CL_DEVICE_MAX_COMPUTE_UNITS: ", pm[[3L]]))
	msg <- c(msg, paste0("    CL_DEVICE_MAX_WORK_GROUP_SIZE: ", pm[[4L]]))
	msg <- c(msg, paste0("    CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: ", pm[[5L]]))
	msg <- c(msg, paste0("    CL_DEVICE_MAX_WORK_ITEM_SIZES: ",
		paste(pm[[6L]], collapse=",")))
	msg <- c(msg, paste0("    CL_DEVICE_LOCAL_MEM_SIZE: ", pm[[7L]]))
	msg <- c(msg, paste0("    CL_DEVICE_ADDRESS_BITS: ", pm[[8L]]))

	.packageEnv$code_clear_memory <- code_clear_memory
	pm <- .Call(ocl_set_kl_clearmem, code_clear_memory)
	msg <- c(msg, paste0("    CL_KERNEL_WORK_GROUP_SIZE: ", pm[1L]))
	msg <- c(msg, paste0("    CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: ", pm[2L]))
	msg <- c(msg, sprintf("    local work size: %d (Dim1), %dx%d (Dim2)",
		pm[3L], pm[4L], pm[4L]))
	local_size_macro <- paste0("#define LOCAL_SIZE_D1    ", pm[3L], "\n",
		"#define LOCAL_SIZE_D2    ", pm[4L])
	.packageEnv$local_size_macro <- local_size_macro

	# support 64-bit floating-point numbers or not
	dev_fp64 <- dev_fp64_ori <- any(grepl("cl_khr_fp64", exts))
	.packageEnv$dev_fp64 <- dev_fp64
	.packageEnv$dev_fp64_ori <- dev_fp64_ori
	if (dev_fp64)
	{
		.packageEnv$code_attempt_f64 <- paste(c(local_size_macro, code_macro,
			code_macro_prec["double"], code_hamm_dist_max["double"], code_atomic_add_f64,
			"#define FIXED_NUM_INT_HAMM    128",
			code_hamming_dist, code_build_calc_prob), collapse="\n")
		# also need cl_khr_int64_base_atomics : enable
		dev_fp64 <- tryCatch({
			.Call(ocl_set_kl_attempt, "build_calc_prob", .packageEnv$code_attempt_f64)
		}, error=function(e) FALSE)
		msg <- c(msg, paste("    atom_cmpxchg (enable cl_khr_int64_base_atomics):",
			ifelse(dev_fp64, "OK", "Failed")))
	}

	msg <- c(msg, paste("GPU device", ifelse(dev_fp64, "supports", "does not support"),
		"double-precision floating-point numbers"))

	# user-defined precision in training
	if (train_prec == "auto") train_prec <- "mixed"
	if (train_prec == "double")
	{
		if (!dev_fp64)
		{
			stop("Unable to use 64-bit floating-point numbers in GPU computing.",
				call.=FALSE)
		}
		f64_build <- TRUE
		msg <- c(msg, "Training uses 64-bit floating-point numbers in GPU")
	} else if (train_prec == "single")
	{
		f64_build <- FALSE
		msg <- c(msg, "Training uses 32-bit floating-point numbers in GPU")
	} else if (train_prec == "mixed")
	{
		f64_build <- FALSE
		msg <- c(msg, "Training uses a mixed precision between half and float in GPU")
	} else if (train_prec == "half")
	{
		f64_build <- FALSE
		msg <- c(msg, "Training uses half precision in GPU")
	} else
		stop("Invalid 'train_prec'.")

	# user-defined precision in prediction
	if (predict_prec == "auto")
	{
		if (has_int64_atom)
		{
			predict_prec <- "double"
			f64_pred <- TRUE
			msg <- c(msg, paste(
				"By default, prediction uses 64-bit floating-point numbers",
				"(since EXTENSION cl_khr_int64_base_atomics: YES)."))
		} else {
			predict_prec <- "single"
			f64_pred <- FALSE
			if (dev_fp64)
			{
				msg <- c(msg, paste(
					"By default, prediction uses 32-bit floating-point numbers in GPU",
					"(since EXTENSION cl_khr_int64_base_atomics: NO)."))
			} else {
				msg <- c(msg, "Prediction uses 32-bit floating-point numbers in GPU.")
			}
		}
	} else if (predict_prec == "double")
	{
		if (!dev_fp64)
		{
			stop("Unable to use 64-bit floating-point numbers in GPU computing.",
				call.=FALSE)
		}
		f64_pred <- TRUE
		msg <- c(msg, "Prediction uses 64-bit floating-point numbers in GPU.")
	} else if (predict_prec == "single")
	{
		f64_pred <- FALSE
		msg <- c(msg, "Prediction uses 32-bit floating-point numbers in GPU.")
	} else
		stop("Invalid 'predict_prec'.")



	## build OpenCL kernels ##

	.packageEnv$flag_build_f64 <- f64_build
	.packageEnv$train_prec <- train_prec
	.packageEnv$flag_pred_f64 <- f64_pred
	.packageEnv$predict_prec <- predict_prec

	.packageEnv$code_build_haplo_match1 <- paste(c(local_size_macro,
		code_macro, code_hamming_dist, code_build_alloc_set, code_build_haplo_match1),
		collapse="\n")
	.packageEnv$code_build_haplo_match2 <- paste(c(local_size_macro,
		code_macro, code_hamming_dist, code_build_alloc_set, code_build_haplo_match2),
		collapse="\n")

	.packageEnv$code_build_calc_prob_int1 <- paste(c(local_size_macro,
		code_macro, code_macro_prec[train_prec], code_hamm_dist_max[train_prec],
		if (f64_build) code_atomic_add_f64 else code_atomic_add_f32,
		"#define FIXED_NUM_INT_HAMM    32",
		code_hamming_dist, code_build_calc_prob), collapse="\n")
	.packageEnv$code_build_calc_prob_int2 <- paste(c(local_size_macro,
		code_macro, code_macro_prec[train_prec], code_hamm_dist_max[train_prec],
		if (f64_build) code_atomic_add_f64 else code_atomic_add_f32,
		"#define FIXED_NUM_INT_HAMM    64",
		code_hamming_dist, code_build_calc_prob), collapse="\n")
	.packageEnv$code_build_calc_prob_int3 <- paste(c(local_size_macro,
		code_macro, code_macro_prec[train_prec], code_hamm_dist_max[train_prec],
		if (f64_build) code_atomic_add_f64 else code_atomic_add_f32,
		"#define FIXED_NUM_INT_HAMM    96",
		code_hamming_dist, code_build_calc_prob), collapse="\n")
	.packageEnv$code_build_calc_prob_int4 <- paste(c(local_size_macro,
		code_macro, code_macro_prec[train_prec], code_hamm_dist_max[train_prec],
		if (f64_build) code_atomic_add_f64 else code_atomic_add_f32,
		"#define FIXED_NUM_INT_HAMM    128",
		code_hamming_dist, code_build_calc_prob), collapse="\n")

	.packageEnv$code_build_calc_oob <- paste(c(local_size_macro,
		c(code_macro, code_macro_prec[train_prec], code_build_calc_oob)), collapse="\n")
	.packageEnv$code_build_calc_ib <- paste(c(local_size_macro,
		code_macro, code_macro_prec[train_prec], code_hamm_dist_max[train_prec],
		code_build_calc_ib), collapse="\n")
	.Call(ocl_set_kl_build, dev_fp64_ori, f64_build, list(
		.packageEnv$code_build_haplo_match1, .packageEnv$code_build_haplo_match2,
		.packageEnv$code_build_calc_prob_int1,
		.packageEnv$code_build_calc_prob_int2,
		.packageEnv$code_build_calc_prob_int3,
		.packageEnv$code_build_calc_prob_int4,
		.packageEnv$code_build_calc_oob, .packageEnv$code_build_calc_ib))

	prec_predict <- ifelse(f64_pred,  "double", "single")
	.packageEnv$code_pred_calc <- paste(c(local_size_macro,
		code_macro, code_macro_prec[predict_prec], code_hamm_dist_max[predict_prec],
		if (f64_pred) code_atomic_add_f64 else code_atomic_add_f32,
		code_hamming_dist, code_pred_calc_prob), collapse="\n")
	.packageEnv$code_pred_sumprob <- paste(c(local_size_macro,
		ifelse(dev_fp64_ori, "#define USE_SUM_DOUBLE", ""),
		code_macro, code_macro_prec[predict_prec], code_pred_calc_sumprob), collapse="\n")
	.packageEnv$code_pred_addprob <- paste(c(
		ifelse(dev_fp64_ori, "#define USE_SUM_DOUBLE", ""),
		code_macro_prec[predict_prec], code_pred_calc_addprob), collapse="\n")
	.Call(ocl_set_kl_predict, f64_pred, .packageEnv$code_pred_calc,
		.packageEnv$code_pred_sumprob, .packageEnv$code_pred_addprob)

	on.exit()
	showmsg(paste(msg, collapse="\n"))
	invisible()
}


# initialize GPU device with given local size
.gpu_set_localsize <- function(sz_d1, sz_d2, verbose=TRUE)
{
	# check
	stopifnot(is.numeric(sz_d1), length(sz_d1)==1L, is.finite(sz_d1), sz_d1>0L)
	stopifnot(is.numeric(sz_d2), length(sz_d2)==1L, is.finite(sz_d2), sz_d2>0L)
	.Call(ocl_set_local_size, sz_d1, sz_d2)

	old_lz_macro <- .packageEnv$local_size_macro
	new_lz_macro <- paste0("#define LOCAL_SIZE_D1    ", sz_d1, "\n",
		"#define LOCAL_SIZE_D2    ", sz_d2)
	nm_lst <- c(
		# training
		"code_build_haplo_match1", "code_build_haplo_match2",
		"code_build_calc_prob_int1", "code_build_calc_prob_int2",
		"code_build_calc_prob_int3", "code_build_calc_prob_int4",
		"code_build_calc_oob", "code_build_calc_ib",
		# prediction
		"code_pred_calc", "code_pred_sumprob", "code_pred_addprob")
	for (nm in nm_lst)
	{
		.packageEnv[[nm]] <- gsub(old_lz_macro, new_lz_macro,
			.packageEnv[[nm]], fixed=TRUE)
	}

	.packageEnv$local_size_macro <- new_lz_macro
	with(.packageEnv, .Call(ocl_set_kl_build, dev_fp64_ori, flag_build_f64, list(
		code_build_haplo_match1, code_build_haplo_match2,
		code_build_calc_prob_int1, code_build_calc_prob_int2,
		code_build_calc_prob_int3, code_build_calc_prob_int4,
		code_build_calc_oob, code_build_calc_ib)))
	with(.packageEnv, .Call(ocl_set_kl_predict, flag_pred_f64,
		code_pred_calc, code_pred_sumprob, code_pred_addprob))

	if (verbose)
	{
		cat(sprintf("using local work size: %d (Dim1), %dx%d (Dim2)\n",
			sz_d1, sz_d2, sz_d2))
	}

	invisible()
}


# initialize the internal GPU methods
hlaGPU_Init <- function(device=NA_integer_,
	train_prec=c("auto", "half", "mixed", "single", "double"),
	predict_prec=c("auto", "single", "double"), verbose=TRUE)
{
	# check
	stopifnot(is.numeric(device), length(device)==1L)
	train_prec <- match.arg(train_prec)
	predict_prec <- match.arg(predict_prec)
	stopifnot(is.logical(verbose), length(verbose)==1L)

	if (is.na(device))
	{
		device <- .packageEnv$gpu_dev_idx
		if (is.null(device)) device <- NA_integer_
	}

	.gpu_init(device, train_prec, predict_prec,
		ifelse(verbose, message, function(x) {}))

	invisible()
}



#######################################################################
# Export stardard R library function(s)
#######################################################################

# get a list of OpenCL devices
.init_dev_list <- function()
{
	.packageEnv$opencl_dev_num <- 0L
	.packageEnv$opencl_dev_num <- .Call(ocl_init_dev_list)
}

# get device information
.get_dev_info <- function(dev_idx) .Call(ocl_dev_info, dev_idx)


# return device index if found
.search_dev <- function(showmsg=message)
{
	# get device info
	info <- vapply(seq_len(.packageEnv$opencl_dev_num), function(idx) {
		s <- .get_dev_info(idx)
		paste0(s[1L], ", ", s[2L])
	}, "")

	# find NVIDIA, AMD and Intel Graphics cards
	i1 <- grep("NVIDIA", info, ignore.case=TRUE)
	i2 <- grep("AMD", info, ignore.case=TRUE)
	i3 <- grep("Intel.*Graphics", info, ignore.case=TRUE)
	ii <- c(i1, i2, i3)
	if (length(ii) <= 0L) ii <- 1L
	ii <- c(ii, setdiff(seq_along(info), ii))

	# build OpenCL kernels
	for (i in ii)
	{
		ok <- tryCatch({
			# check extension
			exts <- .get_dev_info(i)[4L]
			if (!grepl("cl_khr_global_int32_base_atomics", exts))
				stop("Need the OpenCL extension cl_khr_global_int32_base_atomics.")
			# initialize
			.packageEnv$gpu_init_dev_idx <- i
			.gpu_init(i, "auto", "auto", showmsg)
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
	num <- .packageEnv$opencl_dev_num
	msg <- c("", "Available OpenCL device(s):")
	for (i in seq_len(num))
	{
		s <- .get_dev_info(i)
		msg <- c(msg, paste0("    Dev #", i, ": ", s[1L], ", ", s[2L], ", ", s[3L]))
	}
	msg <- c(msg, "")
	packageStartupMessage(paste(msg, collapse="\n"))

	# find NVIDIA, AMD and Intel Graphics cards
	i <- .search_dev(packageStartupMessage)
	if (is.na(i))
		packageStartupMessage("No device supports!")
	else
		packageStartupMessage("")

	TRUE
}

.onLoad <- function(libname, pkgname)
{
	# initialize device list
	.init_dev_list()
	# set procedure pointer
	.packageEnv$gpu_proc_ptr <- .Call(gpu_init_proc, .packageEnv)
	TRUE
}

.onUnload <- function(libpath)
{
	.Call(ocl_release_dev)
}
