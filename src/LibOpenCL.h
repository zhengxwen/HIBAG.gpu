// ===============================================================
//
// HIBAG.gpu R package (GPU-based implementation for the HIBAG package)
// Copyright (C) 2021    Xiuwen Zheng (zhengx@u.washington.edu)
// All rights reserved.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.	 If not, see <http://www.gnu.org/licenses/>.

#ifndef H_LIB_OPENCL
#define H_LIB_OPENCL


#ifdef __APPLE__
#   define CL_SILENCE_DEPRECATION
#	include <OpenCL/opencl.h>
#else
#	include <CL/opencl.h>
#endif

#include <Rdefines.h>


#ifdef __cplusplus
extern "C" {
#endif


// OpenCL device variables
extern cl_context gpu_context;
extern cl_command_queue gpu_command_queue;


/// get information from an OpenCL error code
const char *gpu_error_info(int err);

/// OpenCL call clFinish
void gpu_finish();

/// OpenCL call clReleaseEvent
void gpu_free_events(size_t num_events, const cl_event event_list[]);



// export R functions

extern SEXP ocl_get_dev_list(SEXP Rverbose);


#ifdef __cplusplus
}
#endif

#endif // H_LIB_OPENCL
