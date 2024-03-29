\name{hlaGPU_Init}
\alias{hlaGPU_Init}
\title{
	Initialization of GPU computing
}
\description{
	Initialize the internal GPU methods.
}
\usage{
hlaGPU_Init(device=NA_integer_,
	train_prec=c("auto", "half", "mixed", "single", "double"),
	predict_prec=c("auto", "single", "double"), verbose=TRUE)
}
\arguments{
	\item{device}{a numeric value: 1 for the first device, 2 for the second
	    device, ..., \code{NA_integer_} for using the last selected device or
	    1 at the first call}
	\item{train_prec}{"double": 64-bit floating-point numbers; "single": 32-bit
		floating-point numbers; "mixed": a mixed precision between half and
		float; "half": half precision (16 bits); or "auto", see details}
	\item{predict_prec}{"double": 64-bit floating-point numbers; "single":
		32-bit floating-point numbers; or "auto", see details}
	\item{verbose}{if TRUE, show information}
}
\value{
	None.
}
\details{
	By default (\code{train_prec="auto"}), the model training uses a mixed
precision between half and float, and the prediction uses 64-bit floating-point
numbers in GPU computing if avaliable.
}
\author{Xiuwen Zheng}
\seealso{
	\code{\link{hlaAttrBagging_gpu}}, \code{\link{hlaPredict_gpu}}
}

\examples{
\dontrun{
library(OpenCL)

# use the second GPU device
hlaGPU_Init(2)
}
}

\keyword{HLA}
\keyword{SNP}
\keyword{GPU}
\keyword{genetics}
