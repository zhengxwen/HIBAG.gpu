\name{hlaGPU_Init}
\alias{hlaGPU_Init}
\title{
	Initialization of GPU computing
}
\description{
	Initialize the internal GPU methods.
}
\usage{
hlaGPU_Init(device=NULL, use_double=NA, force=FALSE, verbose=TRUE)
}
\arguments{
	\item{device}{a \code{clDeviceID} object from \link{oclDevices}; or NULL,
		to use the first found device}
	\item{use_double}{TRUE, to use double-precision floating-point numbers;
		FALSE, to use single-precision floating-point numbers; NA, see details}
	\item{force}{if TRUE, force to use double-precision floating-point numbers}
	\item{verbose}{if TRUE, show information}
}
\value{
	None.
}
\details{
	By default (\code{use_double=NA}), model training uses 32-bit floating-point
numbers and prediction uses 64-bit floating-point numbers in GPU computing.
}
\author{Xiuwen Zheng}
\seealso{
	\code{\link{hlaAttrBagging_gpu}}, \code{\link{hlaPredict_gpu}}
}

\examples{
\dontrun{
library(OpenCL)

p = oclPlatforms()
d = oclDevices(p[[1]])

# use the second GPU device
hlaGPU_Init(d[[2L]])
}
}

\keyword{HLA}
\keyword{SNP}
\keyword{GPU}
\keyword{genetics}