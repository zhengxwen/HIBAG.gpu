# HIBAG.gpu -- GPU-based implementation for the HLA genotype imputation method

R package -- a GPU-based extension for the [HIBAG](https://github.com/zhengxwen/HIBAG) package


## Requirements

* [OpenCL](https://cran.r-project.org/web/packages/OpenCL/index.html) R package


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

