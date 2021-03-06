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
extern cl_device_id gpu_device;
extern cl_context gpu_context;
extern cl_command_queue gpu_command_queue;


// OpenCL kernel functions
extern cl_kernel gpu_kl_clear_mem;
extern cl_kernel gpu_kl_build_calc_prob;
extern cl_kernel gpu_kl_build_calc_oob;
extern cl_kernel gpu_kl_build_calc_ib;
extern cl_kernel gpu_kl_pred_calc;
extern cl_kernel gpu_kl_pred_sumprob;
extern cl_kernel gpu_kl_pred_addprob;


// flags for usage of double or single precision
extern bool gpu_f64_flag;
extern bool gpu_f64_build_flag;
extern bool gpu_f64_pred_flag;


/// get information from an OpenCL error code
extern const char *gpu_error_info(int err);

/// OpenCL call clFinish
extern void gpu_finish();

/// OpenCL call clReleaseEvent
extern void gpu_free_events(size_t num_events, const cl_event event_list[]);


/// create memory buffer
extern cl_mem gpu_create_mem(cl_mem_flags flags, size_t size, void *host_ptr,
	const char *fc_nm);



#define GPU_CREATE_MEM(x, flags, size, host_ptr)    \
	x = gpu_create_mem(flags, size, host_ptr, #x)




// export R functions

/// initialize the device list, return the number of devices
extern SEXP ocl_init_dev_list();

/// get information on the device list
extern SEXP ocl_dev_info(SEXP dev_idx);

/// get the parameters on the device
extern SEXP ocl_get_dev_param();

/// select an OpenCL device
extern SEXP ocl_select_dev(SEXP dev_idx);

/// release the previous device
extern SEXP ocl_release_dev();

/// attempt to build a kernel
extern SEXP ocl_set_kl_attempt(SEXP name, SEXP code);

/// set kernel for clear memory
extern SEXP ocl_set_kl_clearmem(SEXP code);

/// set kernels for building the model
extern SEXP ocl_set_kl_build(SEXP f64, SEXP f64_build, SEXP code_prob, SEXP code_oob,
	SEXP code_ib);

/// set kernels for prediction
extern SEXP ocl_set_kl_predict(SEXP f64_pred, SEXP code_calc, SEXP code_sum,
	SEXP code_add);


#ifdef __cplusplus
}
#endif

#endif // H_LIB_OPENCL
