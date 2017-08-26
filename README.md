# HIBAG.gpu -- GPU-based implementation for the HLA genotype imputation method

![GPLv3](http://www.gnu.org/graphics/gplv3-88x31.png)
[GNU General Public License, GPLv3](http://www.gnu.org/copyleft/gpl.html)

R package -- a GPU-based extension for the [HIBAG](https://github.com/zhengxwen/HIBAG) package


## Requirements

* [OpenCL](https://cran.r-project.org/web/packages/OpenCL/index.html) R package


## Performance

Speedup ratios for training HIBAG models:

| CPU (1 core) | CPU (1 core, POPCNT) | 1x NVIDIA Tesla K80 | 1x NVIDIA Tesla M40 | 1x NVIDIA Tesla P100 |
|:------------:|:--------------------:|:-------------------:|:-------------------:|:--------------------:|
| 1 x          | 1.63 x               | 24.3 x              | 35.4 x              | 121.5 x              |

*CPU (1 core), the default installation from Bioconductor supporting SIMD SSE2 instructions, using Intel(R) Xeon(R) CPU E5-2630L @2.40GHz*

*CPU (1 core, POPCNT), optimization with Intel/AMD POPCNT instruction, using Intel(R) Xeon(R) CPU E5-2630L @2.40GHz*

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
## Loading required package: HIBAG
## HIBAG (HLA Genotype Imputation with Attribute Bagging)
## Kernel Version: v1.4
## Supported by Streaming SIMD Extensions (SSE2) [64-bit]
## Loading required package: OpenCL
## Available OpenCL platform(s):
##     NVIDIA CUDA, OpenCL 1.1 CUDA 4.2.1
##         Device #1: NVIDIA Corporation Tesla K80
##
## Using Dev#1: NVIDIA Corporation Tesla K80
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

# SNP predictors within the flanking region on each side
region <- 500   # kb
snpid <- hlaFlankingSNP(HapMap_CEU_Geno$snp.id, HapMap_CEU_Geno$snp.position,
    hla.id, region*1000, assembly="hg19")
length(snpid)  # 275

# training and validation genotypes
train.geno <- hlaGenoSubset(HapMap_CEU_Geno, snp.sel=match(snpid, HapMap_CEU_Geno$snp.id))
summary(train.geno)

# train a HIBAG model
set.seed(100)
model <- hlaAttrBagging_gpu(hla, train.geno, nclassifier=100)
summary(model)
```

```
## Gene: A
## Training dataset: 60 samples X 266 SNPs
##     # of HLA alleles: 14
##     # of individual classifiers: 100
##     total # of SNPs used: 247
##     avg. # of SNPs in an individual classifier: 15.85
##         (sd: 2.43, min: 11, max: 23, median: 16.00)
##     avg. # of haplotypes in an individual classifier: 44.09
##         (sd: 17.05, min: 17, max: 105, median: 41.50)
##     avg. out-of-bag accuracy: 93.65%
##         (sd: 4.97%, min: 78.95%, max: 100.00%, median: 94.22%)
## Matching proportion:
##         Min.     0.1% Qu.       1% Qu.      1st Qu.       Median      3rd Qu. 
## 0.0003151000 0.0003286631 0.0004505229 0.0035640000 0.0097760000 0.0204600000 
##         Max.         Mean           SD 
## 0.4542000000 0.0400800000 0.0994271682 
## Genome assembly: hg19
```

```R
# validation
pred <- hlaPredict_gpu(model, train.geno)
summary(pred)

# compare
(comp <- hlaCompareAllele(hla, pred, allele.limit=model, call.threshold=0)$overall)
```

```
##   total.num.ind crt.num.ind crt.num.haplo   acc.ind acc.haplo call.threshold
## 1            60          59           119 0.9833333 0.9916667              0
##   n.call call.rate
## 1     60         1
```
