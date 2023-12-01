\name{hlaAttrBagging_gpu}
\alias{hlaAttrBagging_gpu}
\alias{hlaAttrBagging_MultiGPU}
\title{
	Build a HIBAG model with a GPU device or multiple devices
}
\description{
	To build a HIBAG model for predicting HLA types with SNP markers.
}
\usage{
hlaAttrBagging_gpu(hla, snp, nclassifier=100L, mtry=c("sqrt", "all", "one"),
	prune=TRUE, na.rm=TRUE, mono.rm=TRUE, maf=NaN, verbose=TRUE,
	verbose.detail=FALSE, use_gpu_predict=TRUE)
hlaAttrBagging_MultiGPU(gpus, hla, snp, auto.save="", nclassifier=100L,
	mtry=c("sqrt", "all", "one"), prune=TRUE, na.rm=TRUE, mono.rm=TRUE,
	maf=NaN, train_prec="auto", use_gpu_predict=TRUE, verbose=TRUE)
}
\arguments{
	\item{hla}{the training HLA types, an object of
		\code{\link{hlaAlleleClass}}}
	\item{snp}{the training SNP genotypes, an object of
		\code{\link{hlaSNPGenoClass}}}
	\item{nclassifier}{the total number of individual classifiers}
	\item{mtry}{a character or a numeric value, the number of variables
		randomly sampled as candidates for each selection. See details}
	\item{prune}{if TRUE, to perform a parsimonious forward variable selection,
		otherwise, exhaustive forward variable selection. See details}
	\item{na.rm}{if TRUE, remove the samples with missing HLA types}
    \item{mono.rm}{if TRUE, remove monomorphic SNPs}
    \item{maf}{MAF threshold for SNP filter, excluding any SNP with MAF < \code{maf}}
	\item{gpus}{a vector of GPU device IDs, \code{gpus} may have duplicated IDs
		to maximize the GPU load}
	\item{auto.save}{specify a autosaved file name for an R object
		(.rda, .RData or .rds file), and the model is auto-saved at most every 10 minutes;
		"", no file saving}
	\item{train_prec}{length-1 character, or a character vector according to \code{gpus}
		for the training precision, see \code{hlaGPU_Init}}
	\item{verbose}{if TRUE, show information}
	\item{verbose.detail}{if TRUE, show more information}
	\item{use_gpu_predict}{if TRUE, use \code{hlaPredict_gpu()} to calculate
	    the matching proportions via GPU, otherwise use \code{hlaPredict()}}
}
\details{
	\code{mtry} (the number of variables randomly sampled as candidates
		for each selection, "sqrt" by default):
	\code{"sqrt"}, using the square root of the total number of candidate SNPs;
	\code{"all"}, using all candidate SNPs;
	\code{"one"}, using one SNP;
	\code{an integer}, specifying the number of candidate SNPs;
	\code{0 < r < 1}, the number of candidate SNPs is
		"r * the total number of SNPs".
	\code{prune}: there is no significant difference on accuracy between
parsimonious and exhaustive forward variable selections. If \code{prune=TRUE},
the searching algorithm performs a parsimonious forward variable selection:
if a new SNP predictor reduces the current out-of-bag accuracy, then it is
removed from the candidate SNP set for future searching. Parsimonious selection
helps to improve the computational efficiency by reducing the searching times
on non-informative SNP markers.
}
\value{
	Return an object of \code{\link{hlaAttrBagClass}}:
	\item{n.samp}{the total number of training samples}
	\item{n.snp}{the total number of candidate SNP predictors}
	\item{sample.id}{the sample IDs}
	\item{snp.id}{the SNP IDs}
	\item{snp.position}{SNP position in basepair}
	\item{snp.allele}{a vector of characters with the format of ``A allele/B allele''}
	\item{snp.allele.freq}{the allele frequencies}
	\item{hla.locus}{the name of HLA locus}
	\item{hla.allele}{the HLA alleles used in the model}
	\item{hla.freq}{the HLA allele frequencies}
	\item{assembly}{the human genome reference, such like "hg19"}
	\item{model}{internal use}

	If \code{auto.save != ""}, return nothing.
}
\references{
	Zheng, X. *et al*. HIBAG--HLA genotype imputation with attribute bagging.
Pharmacogenomics Journal 14, 192-200 (2014).
\url{http://dx.doi.org/10.1038/tpj.2013.18}

	Zheng, X. (2018) Imputation-Based HLA Typing with SNPs in GWAS Studies.
In: Boegel S. (eds) HLA Typing. Methods in Molecular Biology, Vol 1802. Humana Press,
New York, NY. \url{https://doi.org/10.1007/978-1-4939-8546-3_11}
}
\author{Xiuwen Zheng}
\seealso{
	\code{\link{hlaAttrBagging}}, \code{\link{summary.hlaAttrBagClass}},
	\code{\link{hlaPredict}}, \code{\link{hlaPredict_gpu}},
	\code{\link{hlaGPU_Init}}
}

\examples{
# make a "hlaAlleleClass" object
hla.id <- "A"
hla <- hlaAllele(HLA_Type_Table$sample.id,
	H1 = HLA_Type_Table[, paste(hla.id, ".1", sep="")],
	H2 = HLA_Type_Table[, paste(hla.id, ".2", sep="")],
	locus=hla.id, assembly="hg19")

# divide HLA types randomly
set.seed(100)
hlatab <- hlaSplitAllele(hla, train.prop=0.5)
names(hlatab)
# "training"   "validation"
summary(hlatab$training)
summary(hlatab$validation)

# SNP predictors within the flanking region on each side
region <- 500   # kb
snpid <- hlaFlankingSNP(HapMap_CEU_Geno$snp.id, HapMap_CEU_Geno$snp.position,
	hla.id, region*1000, assembly="hg19")
length(snpid)  # 275

# training and validation genotypes
train.geno <- hlaGenoSubset(HapMap_CEU_Geno,
	snp.sel=match(snpid, HapMap_CEU_Geno$snp.id),
	samp.sel=match(hlatab$training$value$sample.id,
	HapMap_CEU_Geno$sample.id))
test.geno <- hlaGenoSubset(HapMap_CEU_Geno,
	samp.sel=match(hlatab$validation$value$sample.id,
	HapMap_CEU_Geno$sample.id))

# train a HIBAG model
set.seed(100)
# please use "nclassifier=100" when you use HIBAG for real data
model <- hlaAttrBagging_gpu(hlatab$training, train.geno, nclassifier=4,
	verbose.detail=TRUE)
summary(model)

# validation
# call hlaPredict or hlaPredict_gpu
pred <- hlaPredict_gpu(model, test.geno)
summary(pred)

# compare
(comp <- hlaCompareAllele(hlatab$validation, pred, allele.limit=model,
	call.threshold=0)$overall)
(comp <- hlaCompareAllele(hlatab$validation, pred, allele.limit=model,
	call.threshold=0.5)$overall)


# save the parameter file
mobj <- hlaModelToObj(model)
save(mobj, file="HIBAG_model.RData")
save(test.geno, file="testgeno.RData")
save(hlatab, file="HLASplit.RData")

# Clear Workspace
hlaClose(model)  # release all resources of model
rm(list = ls())


######################################################################

# NOW, load a HIBAG model from the parameter file
mobj <- get(load("HIBAG_model.RData"))
model <- hlaModelFromObj(mobj)

# validation
test.geno <- get(load("testgeno.RData"))
hlatab <- get(load("HLASplit.RData"))

pred <- hlaPredict_gpu(model, test.geno, type="response")
summary(pred)

# compare
(comp <- hlaCompareAllele(hlatab$validation, pred, allele.limit=model,
	call.threshold=0)$overall)
(comp <- hlaCompareAllele(hlatab$validation, pred, allele.limit=model,
	call.threshold=0.5)$overall)


# delete the temporary files
unlink(c("HIBAG_model.RData", "testgeno.RData", "HLASplit.RData"), force=TRUE)
}

\keyword{HLA}
\keyword{SNP}
\keyword{GPU}
\keyword{genetics}
