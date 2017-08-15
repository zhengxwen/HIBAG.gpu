# HIBAG.gpu -- GPU-based implementation for the HLA genotype imputation method

R package -- a GPU-based extension for the [HIBAG](https://github.com/zhengxwen/HIBAG) package


## Requirements

* [OpenCL](https://cran.r-project.org/web/packages/OpenCL/index.html) R package


## Performance

Speedup:

| CPU (1 core) | CPU + POPCNT (1 core) | NVIDIA Tesla K80 | NVIDIA Tesla M40 | NVIDIA Tesla P100 |
|:------------:|:---------------------:|:----------------:|:----------------:|:-----------------:|
|              | 1                     | 14.9             | 21.7             | 74.6              |

*This work was made possible, in part, through HPC time donated by Microway, Inc. We gratefully acknowledge Microway for providing access to their GPU-accelerated compute cluster (http://www.microway.com/gpu-test-drive/).*


## Installation

* Development version from Github:
```R
library("devtools")
install_github("zhengxwen/HIBAG.gpu")
```
The `install_github()` approach requires that you build from source, i.e. `make` and compilers must be installed on your system -- see the [R FAQ](http://cran.r-project.org/faqs.html) for your operating system; you may also need to install dependencies manually.

* Install the package from the source code:
[download the source code](https://github.com/zhengxwen/HIBAG.gpu/tarball/master)
```sh
wget --no-check-certificate https://github.com/zhengxwen/HIBAG.gpu/tarball/master -O HIBAG.gpu_latest.tar.gz
## or ##
curl -L https://github.com/zhengxwen/HIBAG.gpu/tarball/master/ -o HIBAG.gpu_latest.tar.gz

## Install ##
R CMD INSTALL HIBAG.gpu_latest.tar.gz
```


## Examples

```R
library(HIBAG.gpu)
```

```
## Loading required package: OpenCL
## Loading required package: HIBAG
## HIBAG (HLA Genotype Imputation with Attribute Bagging)
## Kernel Version: v1.4
## Supported by Streaming SIMD Extensions (SSE2 + hardware POPCNT) [64-bit]
## Available OpenCL platform(s):
##     NVIDIA CUDA, OpenCL 1.1 CUDA 4.2.1
##         Device #1: NVIDIA Corporation Tesla K20X
##
## Using NVIDIA Corporation Tesla K20X
## GPU device supports 64-bit floating-point numbers
## By default, training uses 32-bit floating-point numbers and prediction uses 64-bit floating-point numbers in GPU computing.
```

```R
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
model <- hlaAttrBagging_gpu(hlatab$training, train.geno, nclassifier=10)
summary(model)
```

```
## Training dataset: 34 samples X 266 SNPs
##     # of HLA alleles: 14
##     # of individual classifiers: 10
##     total # of SNPs used: 98
##     avg. # of SNPs in an individual classifier: 15.80
##         (sd: 2.94, min: 12, max: 22, median: 15.50)
##     avg. # of haplotypes in an individual classifier: 34.30
##         (sd: 9.68, min: 23, max: 47, median: 34.00)
##     avg. out-of-bag accuracy: 87.37%
##         (sd: 10.43%, min: 66.67%, max: 100.00%, median: 87.12%)
```

```R
# validation
pred <- hlaPredict_gpu(model, test.geno)
summary(pred)

# compare
(comp <- hlaCompareAllele(hlatab$validation, pred, allele.limit=model,
    call.threshold=0)$overall)
```

```
##   total.num.ind crt.num.ind crt.num.haplo   acc.ind acc.haplo call.threshold
## 1            26          22            48 0.8461538 0.9230769              0
##   n.call call.rate
## 1     26         1
```
