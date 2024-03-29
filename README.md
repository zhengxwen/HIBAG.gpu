# HIBAG.gpu – GPU-based implementation for the HLA genotype imputation method

![GPLv3](http://www.gnu.org/graphics/gplv3-88x31.png)
[GNU General Public License, GPLv3](http://www.gnu.org/copyleft/gpl.html)

R package – a GPU-based extension for the [HIBAG](https://github.com/zhengxwen/HIBAG) package


## Requirements

* OpenCL C library and header files ([https://www.khronos.org](https://www.khronos.org))

* [HIBAG](https://github.com/zhengxwen/HIBAG) R package (≥ v1.28.0)


## News

**Changes in v0.99.1 (Y2023):**

* the R global option variable 'HIBAG_GPU_INIT_TRAIN_PREC' can be set before loading the HIBAG.gpu package, see [NEWS](./NEWS)

* a configure file for macOS and Linux

**Changes in v0.99.0 (Y2021):**

* implementation for using half and mixed precisions

* a new function `hlaAttrBagging_MultiGPU()` to leverage multiple GPU devices

**v0.9.0 (Y2018):**

* initial GPU-based implementation for the HIBAG algorithm


## Benchmarks for model training

### 1) Speedup factors using small training sets (~1,000 samples)

| CPU / GPU              | Precision: half | mixed  | single | double |
|:-----------------------|----------------:|-------:|-------:|-------:|
| CPU (AVX2, 1 thread)   | ---             | ---    | ---    | 1      |
| CPU (AVX2, 20 threads) | ---             | ---    | ---    | 14.7   |
| 1x NVIDIA GTX 1080Ti   | 48.2            | 40.5   | 36.5   | ---    |
| 1x NVIDIA Tesla T4     | 72.9            | 64.5   | 57.1   | ---    |
| 1x NVIDIA Tesla V100   | 82.7            | 73.5   | 66.9   | 20.7   |
| 1x NVIDIA Tesla A100   | 133.5           | 114.4  | 102.1  | 23.0   |
| Apple M1 Max (32-core GPU) | 91.5        | 64.6   | 57.3   |        |

### 2) Speedup factors using medium training sets (~5,000 samples)

| CPU / GPU              | Precision: half | mixed  | single | double |
|:-----------------------|----------------:|-------:|-------:|-------:|
| CPU (AVX2, 1 thread)   | ---             | ---    | ---    | 1      |
| CPU (AVX2, 20 threads) | ---             | ---    | ---    | 17.5   |
| 1x NVIDIA GTX 1080Ti   | 88.7            | 65.8   | 58.1   | ---    |
| 1x NVIDIA Tesla T4     | 108.1           | 76.3   | 63.8   | ---    |
| 1x NVIDIA Tesla V100   | 135.7           | 107.5  | 99.0   | 25.3   |
| 1x NVIDIA Tesla A100   | 149.0           | 114.6  | 107.8  |        |
| Apple M1 Max (32-core GPU) | 138.8       | 93.8   | 78.5   |        |

*† ‘mixed’ is a mixed precision between half and single*

*† models built on HLA-A, -B, -C, -DRB1 using HIBAG v1.26.1 and HIBAG.gpu v0.99.0, and the averages are reported*

*† CPU (AVX2, 1/20 threads), optimization with Intel AVX2 instruction, using Intel(R) Xeon(R) Gold 6248 2.50GHz, 20 cores (Cascade Lake)*

*† This work was made possible, in part, through HPC time donated by Microway, Inc. We gratefully acknowledge Microway for providing access to their GPU-accelerated compute cluster (http://www.microway.com/gpu-test-drive/).*


## Citation

1. Zheng, X. *et al*. HIBAG-HLA genotype imputation with attribute bagging. *Pharmacogenomics Journal* 14, 192-200 (2014).
[http://dx.doi.org/10.1038/tpj.2013.18](http://dx.doi.org/10.1038/tpj.2013.18)

2. Zheng, X. (2018) Imputation-Based HLA Typing with SNPs in GWAS Studies. In: Boegel S. (eds) HLA Typing. Methods in Molecular Biology, Vol 1802. Humana Press, New York, NY. [https://doi.org/10.1007/978-1-4939-8546-3_11](https://doi.org/10.1007/978-1-4939-8546-3_11)


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

## Loading required package: HIBAG
## HIBAG (HLA Genotype Imputation with Attribute Bagging)
## Kernel Version: v1.5 (64-bit, AVX2)
## 
## Available OpenCL device(s):
##     Dev #1: AMD, AMD Radeon Pro 560X Compute Engine, OpenCL 1.2
## 
## Using Device #1: AMD, AMD Radeon Pro 560X Compute Engine
##     Device Version: OpenCL 1.2 (>= v1.2: YES)
##     Driver Version: 1.2 (Jan 12 2021 22:17:03)
##     EXTENSION cl_khr_global_int32_base_atomics: YES
##     EXTENSION cl_khr_fp64: YES
##     EXTENSION cl_khr_int64_base_atomics: NO
##     EXTENSION cl_khr_global_int64_base_atomics: NO
##     CL_DEVICE_GLOBAL_MEM_SIZE: 4,294,967,296
##     CL_DEVICE_MAX_MEM_ALLOC_SIZE: 1,073,741,824
##     CL_DEVICE_MAX_COMPUTE_UNITS: 16
##     CL_DEVICE_MAX_WORK_GROUP_SIZE: 256
##     CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: 3
##     CL_DEVICE_MAX_WORK_ITEM_SIZES: 256,256,256
##     CL_DEVICE_LOCAL_MEM_SIZE: 32768
##     CL_DEVICE_ADDRESS_BITS: 32
##     CL_KERNEL_WORK_GROUP_SIZE: 256
##     CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: 64
##     local work size: 256 (Dim1), 16x16 (Dim2)
##     atom_cmpxchg (enable cl_khr_int64_base_atomics): OK
## GPU device supports double-precision floating-point numbers
## Training uses a mixed precision between half and float in GPU
## By default, prediction uses 32-bit floating-point numbers in GPU (since EXTENSION cl_khr_int64_base_atomics: NO).
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
length(snpid)

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
##     total # of SNPs used: 239
##     avg. # of SNPs in an individual classifier: 15.60
##         (sd: 2.73, min: 10, max: 24, median: 16.00)
##     avg. # of haplotypes in an individual classifier: 40.44
##         (sd: 17.46, min: 18, max: 105, median: 36.00)
##     avg. out-of-bag accuracy: 93.00%
##         (sd: 4.94%, min: 78.95%, max: 100.00%, median: 93.48%)
## Matching proportion:
##         Min.     0.1% Qu.       1% Qu.      1st Qu.       Median      3rd Qu.
## 0.0004657777 0.0004773751 0.0005817518 0.0040887403 0.0112166087 0.0282807795
##          Max.         Mean           SD
##  0.4263384035 0.0393261556 0.0919757944
## Genome assembly: hg19
```

```R
# validation
pred <- hlaPredict_gpu(model, train.geno)
summary(pred)

# compare
(comp <- hlaCompareAllele(hla, pred, call.threshold=0)$overall)
```

```
## total.num.ind crt.num.ind crt.num.haplo   acc.ind acc.haplo call.threshold
##            60          59           119 0.9833333 0.9916667              0
##  n.call call.rate
##      60         1
```
