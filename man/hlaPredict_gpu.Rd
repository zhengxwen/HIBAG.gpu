\name{hlaPredict_gpu}
\alias{hlaPredict_gpu}
\title{
    HIBAG model prediction with a GPU device
}
\description{
    To predict HLA type based on a HIBAG model with a GPU device.
}
\usage{
hlaPredict_gpu(model, snp, type=c("response+dosage", "response", "prob", "response+prob"),
	vote=c("prob", "majority"), allele.check=TRUE,
	match.type=c("Position", "Pos+Allele", "RefSNP+Position", "RefSNP"),
	same.strand=FALSE, verbose=TRUE, verbose.match=TRUE)
}
\arguments{
	\item{model}{a model of \code{\link{hlaAttrBagClass}}}
	\item{snp}{a genotypic object of \code{\link{hlaSNPGenoClass}}}
	\item{type}{"response+dosage": return the best-guess types and dosages for
		each allele (by default); "response": return the best-guess types with
		its posterior probability; "prob": return all posterior probabilities;
		"response+prob": return the best-guess and all posterior probabilities}
	\item{vote}{\code{"prob"} (default behavior) -- make a prediction based on
		the averaged posterior probabilities from all individual classifiers;
		\code{"majority"} -- majority voting from all individual
		classifiers, where each classifier votes for an HLA type}
	\item{allele.check}{if \code{TRUE}, check and then switch allele pairs
		if needed}
	\item{match.type}{\code{"RefSNP+Position"} (by default) -- using both of
		RefSNP IDs and positions; \code{"RefSNP"} -- using RefSNP IDs only;
		\code{"Position"} -- using positions only}
	\item{same.strand}{\code{TRUE} assuming alleles are on the same strand
		(e.g., forward strand); otherwise, \code{FALSE} not assuming whether
		on the same strand or not}
	\item{verbose}{if TRUE, show information}
	\item{verbose.match}{if TRUE, show missing SNP proportions for different
		\code{match.type}}
}
\value{
	Return a \code{\link{hlaAlleleClass}} object with posterior probabilities
of predicted HLA types, or a matrix of pairwise possible HLA types with all
posterior probabilities. If \code{type = "response+prob"}, return a
\code{\link{hlaAlleleClass}} object with a matrix of \code{postprob} for the
probabilities of all pairs of alleles.
	If a probability matrix is returned, \code{colnames} is \code{sample.id}
and \code{rownames} is an unordered pair of HLA alleles.
}
\details{
	If more than 50\% of SNP predictors are missing, a warning will be given.

	When \code{match.type="RefSNP+Position"}, the matching of SNPs requires
both RefSNP IDs and positions. A lower missing fraction maybe gained by
matching RefSNP IDs or positions only. Call
	\code{predict(..., match.type="RefSNP")} or
	\code{predict(..., match.type="Position")} for this purpose.
It might be safe to assume that the SNPs with the same positions on the same
genome reference (e.g., hg19) are the same variant albeit the different RefSNP
IDs. Any concern about SNP mismatching should be emailed to the genotyping
platform provider.
}
\author{Xiuwen Zheng}
\seealso{
	\code{\link{hlaPredict}},
	\code{\link{hlaAttrBagging}}, \code{\link{hlaAttrBagging_gpu}},
	\code{\link{hlaAllele}}, \code{\link{hlaCompareAllele}}
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
model <- hlaAttrBagging_gpu(hlatab$training, train.geno, nclassifier=4,
	verbose.detail=TRUE)
summary(model)

# validation
pred <- hlaPredict_gpu(model, test.geno)
# compare
(comp <- hlaCompareAllele(hlatab$validation, pred, allele.limit=model,
	call.threshold=0)$overall)
(comp <- hlaCompareAllele(hlatab$validation, pred, allele.limit=model,
	call.threshold=0.5)$overall)
}

\keyword{HLA}
\keyword{SNP}
\keyword{GPU}
\keyword{genetics}
