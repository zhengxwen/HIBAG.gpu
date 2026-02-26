// ===============================================================
//
// HIBAG.gpu R package (GPU-based implementation for the HIBAG package)
// Copyright (C) 2021-2026    Xiuwen Zheng (zhengx@u.washington.edu)
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
#ifdef length
#   undef length
#endif


#ifdef __cplusplus
extern "C" {
#endif


// OpenCL device variables
extern cl_device_id gpu_device;
extern bool gpu_device_OpenCL_1_2;
extern cl_context gpu_context;
extern cl_command_queue gpu_command_queue;


// OpenCL kernel functions
extern cl_kernel gpu_kl_clear_mem;
extern cl_kernel gpu_kl_build_haplo_match1;
extern cl_kernel gpu_kl_build_haplo_match2;
extern cl_kernel gpu_kl_build_calc_prob_int1;
extern cl_kernel gpu_kl_build_calc_prob_int2;
extern cl_kernel gpu_kl_build_calc_prob_int3;
extern cl_kernel gpu_kl_build_calc_prob_int4;
extern cl_kernel gpu_kl_build_calc_oob;
extern cl_kernel gpu_kl_build_calc_ib;
extern cl_kernel gpu_kl_pred_calc;
extern cl_kernel gpu_kl_pred_sumprob;
extern cl_kernel gpu_kl_pred_addprob;


// OpenCL memory buffer
extern cl_mem mem_rare_freq_f32;   // float[]
extern cl_mem mem_rare_freq_f64;   // double[]
extern cl_mem mem_build_hla_idx_map;  // int[], an index according to a pair of alleles
extern cl_mem mem_build_idx_oob;   // parameters, int[]
extern cl_mem mem_build_idx_ib;    // parameters, int[]
extern cl_mem mem_build_haplo_idx; // parameters, int[NumHLA+1]
extern cl_mem mem_build_output;    // int[], float[], or double[]
extern cl_mem mem_snpgeno;         // SNP genotypes, TGenotype[]
extern cl_mem mem_haplo_list;      // haplotype list, THaplotype[]
extern cl_mem mem_prob_buffer;     // double[nHLA*(nHLA+1)/2][# of samples]
extern cl_mem mem_pred_haplo_num;  // num of haplotypes and SNPs for each classifier: int[][4]
extern cl_mem mem_pred_weight;     // classifier weight



// flags for usage of double or single precision
extern bool gpu_f64_flag;
extern bool gpu_f64_build_flag;
extern bool gpu_f64_pred_flag;


// used for work-group size (1-dim and 2-dim)
extern size_t gpu_local_size_d1;
extern size_t gpu_local_size_d2;


// verbose in OpenCL implementation
extern bool ocl_verbose;



/// get a string from an integer
extern const char *long_to_str(long long sz);

/// get information from an OpenCL error code
extern const char *gpu_error_info(int err);

/// OpenCL call clFinish
extern void gpu_finish();

/// OpenCL call clReleaseEvent
extern void gpu_free_events(size_t num_events, const cl_event event_list[]);


/// create memory buffer
extern cl_mem gpu_create_mem(cl_mem_flags flags, size_t size, void *host_ptr,
	const char *fc_nm);

/// release memory buffer
extern void gpu_free_mem(cl_mem mem, const char *fc_nm);

/// write memory buffer
extern cl_event gpu_write_mem(cl_mem buffer, bool blocking, size_t size, const void *ptr,
	const char *fc_nm);

/// read memory buffer
extern void gpu_read_mem(cl_mem buffer, size_t offset, size_t size, void *ptr,
	const char *fc_nm);

/// set kernel argument
extern void gpu_setarg(cl_kernel kernel, int arg_idx, size_t arg_size,
	const void *arg_value, const char *fc_nm);

/// copy buffer memory
extern void gpu_copy_buffer(cl_mem src_buf, cl_mem dst_buf, size_t src_offset,
	size_t dst_offset, size_t cb, cl_uint num_events_in_wait_list,
	const cl_event *event_wait_list, cl_event *event, const char *s_nm, const char *d_nm);


// define MARCO

#define GPU_CREATE_MEM(x, flags, size, p)    \
	x = gpu_create_mem(flags, size, p, #x)
#define GPU_CREATE_MEM_V(x, flags, size, host_ptr)    \
	{ \
		size_t sz = size; \
		if (ocl_verbose) \
			Rprintf("    allocating %s bytes in GPU  ", long_to_str(sz)); \
		x = gpu_create_mem(flags, sz, host_ptr, #x); \
		if (ocl_verbose) \
			Rprintf("[OK]\n"); \
	}
#define GPU_FREE_MEM(x)    gpu_free_mem(x, #x)

#define GPU_WRITE_MEM(x, size, ptr)      gpu_write_mem(x, true, size, ptr, #x)
#define GPU_WRITE_EVENT(x, size, ptr)    gpu_write_mem(x, false, size, ptr, #x)

#define GPU_READ_MEM(x, offset, size, ptr)  gpu_read_mem(x, offset, size, ptr, #x)

#define GPU_SETARG(kernel, i, x)           gpu_setarg(kernel, i, sizeof(x), &x, #kernel)
#define GPU_SETARG_LOCAL(kernel, i, sz)    gpu_setarg(kernel, i, sz, NULL, #kernel)

#define GPU_COPY_BUFFER(s_buf, d_buf, s_offset, d_offset, cb, n_e, e_lst, out_e)   \
	gpu_copy_buffer(s_buf, d_buf, s_offset, d_offset, cb, n_e, e_lst, out_e, #s_buf, #d_buf)




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
extern SEXP ocl_set_kl_build(SEXP f64, SEXP f64_build, SEXP codes);

/// set kernels for prediction
extern SEXP ocl_set_kl_predict(SEXP f64_pred, SEXP code_calc, SEXP code_sum,
	SEXP code_add);


#ifdef __cplusplus
}
#endif

#endif // H_LIB_OPENCL
