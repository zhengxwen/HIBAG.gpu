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


#include "LibOpenCL.h"

#include <string.h>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

#define USE_RINTERNALS 1
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

// Defined in HIBAG/install/LibHLA_ext.h
#define HIBAG_STRUCTURE_HEAD_ONLY
#include <LibHLA_ext.h>


extern "C"
{
// OpenCL device list
static std::vector<cl_device_id> gpu_dev_list;

// OpenCL device variables
cl_device_id gpu_device = 0;
cl_context gpu_context = NULL;
cl_command_queue gpu_command_queue = NULL;


// OpenCL kernel functions
cl_kernel gpu_kl_clear_mem = NULL;

// OpenCL kernel functions
cl_kernel gpu_kl_build_calc_prob = NULL;
cl_kernel gpu_kl_build_calc_oob = NULL;
cl_kernel gpu_kl_build_calc_ib = NULL;
cl_kernel gpu_kl_pred_calc = NULL;
cl_kernel gpu_kl_pred_sumprob = NULL;
cl_kernel gpu_kl_pred_addprob = NULL;


// OpenCL memory buffer
cl_mem mem_rare_freq_f32 = NULL;
cl_mem mem_rare_freq_f64 = NULL;


// flags for usage of double or single precision
bool gpu_f64_flag = false;
bool gpu_f64_build_flag = false;
bool gpu_f64_pred_flag  = false;


// OpenCL kernel function names
static const char *kl_nm_clear_mem = "clear_memory";
static const char *kl_nm_build_calc_prob = "build_calc_prob";
static const char *kl_nm_build_calc_oob = "build_calc_oob";
static const char *kl_nm_build_calc_ib = "build_calc_ib";
static const char *kl_nm_pred_calc_prob = "pred_calc_prob";
static const char *kl_nm_pred_calc_sumprob = "pred_calc_sumprob";
static const char *kl_nm_pred_calc_addprob = "pred_calc_addprob";



// ===================================================================== //

/// get information from an OpenCL error code
const char *gpu_error_info(int err)
{
	#define ERR_RET(s)    case s: return #s;
	switch (err)
	{
		ERR_RET(CL_SUCCESS)
		ERR_RET(CL_DEVICE_NOT_FOUND)
		ERR_RET(CL_DEVICE_NOT_AVAILABLE)
		ERR_RET(CL_COMPILER_NOT_AVAILABLE)
		ERR_RET(CL_MEM_OBJECT_ALLOCATION_FAILURE)
		ERR_RET(CL_OUT_OF_RESOURCES)
		ERR_RET(CL_OUT_OF_HOST_MEMORY)
		ERR_RET(CL_PROFILING_INFO_NOT_AVAILABLE)
		ERR_RET(CL_MEM_COPY_OVERLAP)
		ERR_RET(CL_IMAGE_FORMAT_MISMATCH)
		ERR_RET(CL_IMAGE_FORMAT_NOT_SUPPORTED)
		ERR_RET(CL_BUILD_PROGRAM_FAILURE)
		ERR_RET(CL_MAP_FAILURE)
		ERR_RET(CL_INVALID_VALUE)
		ERR_RET(CL_INVALID_DEVICE_TYPE)
		ERR_RET(CL_INVALID_PLATFORM)
		ERR_RET(CL_INVALID_DEVICE)
		ERR_RET(CL_INVALID_CONTEXT)
		ERR_RET(CL_INVALID_QUEUE_PROPERTIES)
		ERR_RET(CL_INVALID_COMMAND_QUEUE)
		ERR_RET(CL_INVALID_HOST_PTR)
		ERR_RET(CL_INVALID_MEM_OBJECT)
		ERR_RET(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR)
		ERR_RET(CL_INVALID_IMAGE_SIZE)
		ERR_RET(CL_INVALID_SAMPLER)
		ERR_RET(CL_INVALID_BINARY)
		ERR_RET(CL_INVALID_BUILD_OPTIONS)
		ERR_RET(CL_INVALID_PROGRAM)
		ERR_RET(CL_INVALID_PROGRAM_EXECUTABLE)
		ERR_RET(CL_INVALID_KERNEL_NAME)
		ERR_RET(CL_INVALID_KERNEL_DEFINITION)
		ERR_RET(CL_INVALID_KERNEL)
		ERR_RET(CL_INVALID_ARG_INDEX)
		ERR_RET(CL_INVALID_ARG_VALUE)
		ERR_RET(CL_INVALID_ARG_SIZE)
		ERR_RET(CL_INVALID_KERNEL_ARGS)
		ERR_RET(CL_INVALID_WORK_DIMENSION)
		ERR_RET(CL_INVALID_WORK_GROUP_SIZE)
		ERR_RET(CL_INVALID_WORK_ITEM_SIZE)
		ERR_RET(CL_INVALID_GLOBAL_OFFSET)
		ERR_RET(CL_INVALID_EVENT_WAIT_LIST)
		ERR_RET(CL_INVALID_EVENT)
		ERR_RET(CL_INVALID_OPERATION)
		ERR_RET(CL_INVALID_GL_OBJECT)
		ERR_RET(CL_INVALID_BUFFER_SIZE)
		ERR_RET(CL_INVALID_MIP_LEVEL)
		ERR_RET(CL_INVALID_GLOBAL_WORK_SIZE)

	#ifdef CL_VERSION_1_1
		ERR_RET(CL_MISALIGNED_SUB_BUFFER_OFFSET)
		ERR_RET(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST)
		ERR_RET(CL_INVALID_PROPERTY)
	#endif
	#ifdef CL_VERSION_1_2
		ERR_RET(CL_COMPILE_PROGRAM_FAILURE)
		ERR_RET(CL_LINKER_NOT_AVAILABLE)
		ERR_RET(CL_LINK_PROGRAM_FAILURE)
		ERR_RET(CL_DEVICE_PARTITION_FAILED)
		ERR_RET(CL_KERNEL_ARG_INFO_NOT_AVAILABLE)
		ERR_RET(CL_INVALID_IMAGE_DESCRIPTOR)
		ERR_RET(CL_INVALID_COMPILER_OPTIONS)
		ERR_RET(CL_INVALID_LINKER_OPTIONS)
		ERR_RET(CL_INVALID_DEVICE_PARTITION_COUNT)
	#endif
	#ifdef CL_VERSION_2_0
		ERR_RET(CL_INVALID_PIPE_SIZE)
		ERR_RET(CL_INVALID_DEVICE_QUEUE)
	#endif
	#ifdef CL_VERSION_2_2
		ERR_RET(CL_INVALID_SPEC_ID)
		ERR_RET(CL_MAX_SIZE_RESTRICTION_EXCEEDED)
	#endif
	}
	return "UNKNOWN";
	#undef ERR_RET
}


/// OpenCL error message
static const char *gpu_err_msg(int err, const char *txt, const char *var=NULL)
{
	static char buf[1024];
	const char *info = gpu_error_info(err);
	if (var)
	{
		sprintf(buf, "%s '%s' (error: %d, %s).", txt, var, err, info);
	} else {
		sprintf(buf, "%s (error: %d, %s).", txt, err, info);
	}
	return buf;
}


/// OpenCL call clFinish
void gpu_finish()
{
	cl_int err = clFinish(gpu_command_queue);
	if (err != CL_SUCCESS)
		throw gpu_err_msg(err, "Failed to call clFinish()");
}

/// OpenCL call clReleaseEvent
void gpu_free_events(size_t num_events, const cl_event event_list[])
{
	for (size_t i=0; i < num_events; i++)
		clReleaseEvent(event_list[i]);
}


static cl_kernel build_kernel(SEXP code, const char *name)
{
	static const char *fc_create_prog_src = "clCreateProgramWithSource";
	static const char *fc_build_program = "clBuildProgram";
	static const char *fc_create_kernel = "clCreateKernel";
	static const char *err_info = "Failed to build a kernel '%s'";

	const int sn = Rf_length(code);
	const char **cptr = (const char **)malloc(sizeof(char*) * sn);
	if (cptr == NULL)
		Rf_error("Out of memory in build_kernel()");
	for (int i = 0; i < sn; i++)
	    cptr[i] = CHAR(STRING_ELT(code, i));

	cl_int err;
	cl_program program = clCreateProgramWithSource(gpu_context, sn, cptr, NULL, &err);
	free(cptr);
	if (!program)
		Rf_error(gpu_err_msg(err, err_info, fc_create_prog_src), name);

	// build ...
	err = clBuildProgram(program, 1, &gpu_device, NULL, NULL, NULL);

	size_t log_len = 0;
	if (clGetProgramBuildInfo(program, gpu_device, CL_PROGRAM_BUILD_LOG, 0, NULL,
		&log_len)==CL_SUCCESS && log_len > 1)
	{
		char *buffer = (char*)malloc(log_len);
		if (buffer)
		{
			if (clGetProgramBuildInfo(program, gpu_device, CL_PROGRAM_BUILD_LOG, log_len,
				buffer, NULL) == CL_SUCCESS)
			{
				R_ShowMessage(buffer);
			} else {
				R_ShowMessage("Could not obtain build log");
            }
			free(buffer);
		} else
			R_ShowMessage("Out of memory in build_kernel() for log buffer");
	}
	if (err != CL_SUCCESS)
	{
		clReleaseProgram(program);
		Rf_error(gpu_err_msg(err, err_info, fc_build_program), name);
	}

	cl_kernel kl = clCreateKernel(program, name, &err);
    clReleaseProgram(program);
    if (!kl || err!=CL_SUCCESS)
		Rf_error(gpu_err_msg(err, err_info, fc_create_kernel), name);

	return kl;
}

static int get_kernel_param(cl_device_id dev, cl_kernel kernel,
	cl_kernel_work_group_info param)
{
	size_t n = 0;
	cl_int err = clGetKernelWorkGroupInfo(kernel, dev, param, sizeof(n), &n, NULL);
	if (err != CL_SUCCESS) return NA_INTEGER;
	return n;
}


cl_mem gpu_create_mem(cl_mem_flags flags, size_t size, void *host_ptr,
	const char *fc_nm)
{
	cl_int err;
	cl_mem mem = clCreateBuffer(gpu_context, flags, size, host_ptr, &err);
	if (!mem)
	{
		if (fc_nm)
		{
			Rf_error("Failed to create memory buffer (%lld bytes) '%s' (error: %d, %s).",
				(long long)size, fc_nm, err, gpu_error_info(err));
		} else {
			Rf_error("Failed to create memory buffer (%lld bytes) (error: %d, %s).",
				(long long)size, err, gpu_error_info(err));
		}
	}
	return mem;
}



// ===================================================================== //

static std::vector<std::string> tmp_ss;

static void check_dev_idx(int idx)
{
	if (idx < 1 || idx > gpu_dev_list.size())
	{
		Rf_error("The selected device index should be between 1 and %d.",
			(int)gpu_dev_list.size());
	}	
}

static std::string get_dev_info_str(cl_device_id dev, cl_device_info p)
{
	static const char *fc_get_device_info = "clGetDeviceInfo";
	static const char *err_info = "Failed to get device information";

	size_t ret_n = 0;
	cl_int err = clGetDeviceInfo(dev, p, 0, NULL, &ret_n);
	if (err != CL_SUCCESS)
		Rf_error(gpu_err_msg(err, err_info, fc_get_device_info));

	std::string s(ret_n, 0);
	err = clGetDeviceInfo(dev, p, s.size(), &s[0], NULL);
	if (err != CL_SUCCESS)
		Rf_error(gpu_err_msg(err, err_info, fc_get_device_info));
	return s;
}


/// initialize the device list, return the number of devices
SEXP ocl_init_dev_list()
{
	static const char *fc_get_platform = "clGetPlatformIDs";
	static const char *fc_get_device   = "clGetDeviceIDs";
	static const char *err_info = "Failed to get a list of devices";

	// clear the list
	gpu_dev_list.clear();

	// get a list of platforms
	cl_uint num_platforms = 0;
	cl_int err = clGetPlatformIDs(0, NULL, &num_platforms);
	if (err != CL_SUCCESS)
		Rf_error(gpu_err_msg(err, err_info, fc_get_platform));
	cl_platform_id *platforms =
		(cl_platform_id*)malloc(num_platforms*sizeof(cl_platform_id));
	err = clGetPlatformIDs(num_platforms, platforms, NULL);
	if (err != CL_SUCCESS)
	{
		free(platforms);
		Rf_error(gpu_err_msg(err, err_info, fc_get_platform));
	}

	// get a list of devices for each platform
	for (cl_uint i=0; i < num_platforms; i++)
	{
		cl_platform_id platform = platforms[i];
		// get devices using this platform
		cl_uint num_dev = 0;
		err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_dev);
		if (err!=CL_SUCCESS && err!=CL_DEVICE_NOT_FOUND)
		{
			free(platforms);
			Rf_error(gpu_err_msg(err, err_info, fc_get_device));
		}
		if (num_dev > 0)
		{
			cl_device_id *dev = (cl_device_id*)malloc(sizeof(cl_device_id)*num_dev);
			err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_dev, dev, NULL);
			if (err != CL_SUCCESS)
			{
				free(dev);
				free(platforms);
				error(gpu_err_msg(err, err_info, fc_get_device));
			}
			for (cl_uint j=0; j < num_dev; j++)
				gpu_dev_list.push_back(dev[j]);
			free(dev);
		}
	}

	// finally
	free(platforms);
	return Rf_ScalarInteger(gpu_dev_list.size());
}


/// get information on the device list
extern SEXP ocl_dev_info(SEXP dev_idx)
{
	const int idx = Rf_asInteger(dev_idx);
	check_dev_idx(idx);
	cl_device_id dev = gpu_dev_list[idx-1];

	// get device information
	const int n_ds = 6;
	static cl_device_info ds[n_ds] = {
		CL_DEVICE_VENDOR, CL_DEVICE_NAME, CL_DEVICE_VERSION, CL_DEVICE_EXTENSIONS,
		CL_DEVICE_PROFILE, CL_DRIVER_VERSION
	};
	tmp_ss.clear();
	for (size_t i=0; i < n_ds; i++)
		tmp_ss.push_back(get_dev_info_str(dev, ds[i]));

	{
		std::string s = get_dev_info_str(dev, CL_DEVICE_TYPE);
		cl_device_type dt = *((cl_device_type*)&s[0]);
		switch (dt)
		{
			case CL_DEVICE_TYPE_CPU:
				tmp_ss.push_back("CL_DEVICE_TYPE_CPU"); break;
			case CL_DEVICE_TYPE_GPU:
				tmp_ss.push_back("CL_DEVICE_TYPE_GPU"); break;
			case CL_DEVICE_TYPE_ACCELERATOR:
				tmp_ss.push_back("CL_DEVICE_TYPE_ACCELERATOR"); break;
			case CL_DEVICE_TYPE_DEFAULT:
				tmp_ss.push_back("CL_DEVICE_TYPE_DEFAULT"); break;
			default:
				tmp_ss.push_back("Unknown Device Type");
		}
	}

	// output
	SEXP rv_ans = PROTECT(NEW_CHARACTER(tmp_ss.size()));
	for (size_t i=0; i < tmp_ss.size(); i++)
		SET_STRING_ELT(rv_ans, i, mkChar(tmp_ss[i].c_str()));
	UNPROTECT(1);
	return rv_ans;
}


/// get the parameters on the device
extern SEXP ocl_get_dev_param()
{
	#define GET_DEV_INFO(TYPE, PARAM, VAR)    { \
		TYPE v; \
		cl_int err = clGetDeviceInfo(gpu_device, PARAM, sizeof(v), &v, NULL); \
		if (err == CL_SUCCESS) VAR = v; \
	}

	double gl_mem_sz = R_NaN;
	GET_DEV_INFO(cl_ulong, CL_DEVICE_GLOBAL_MEM_SIZE, gl_mem_sz)
	double gl_mem_alloc_sz = R_NaN;
	GET_DEV_INFO(cl_ulong, CL_DEVICE_MAX_MEM_ALLOC_SIZE, gl_mem_alloc_sz)
	int gl_n_unit = NA_INTEGER;
	GET_DEV_INFO(cl_uint, CL_DEVICE_MAX_COMPUTE_UNITS, gl_n_unit)
	int gl_max_worksize = NA_INTEGER;
	GET_DEV_INFO(size_t, CL_DEVICE_MAX_WORK_GROUP_SIZE, gl_max_worksize)
	int gl_max_workdim = NA_INTEGER;
	GET_DEV_INFO(cl_uint, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, gl_max_workdim)
	int gl_local_mem_sz = NA_INTEGER;
	GET_DEV_INFO(cl_ulong, CL_DEVICE_LOCAL_MEM_SIZE, gl_local_mem_sz)
	int gl_addr_bits = NA_INTEGER;
	GET_DEV_INFO(cl_uint, CL_DEVICE_ADDRESS_BITS, gl_addr_bits)

	size_t gl_max_work_item_sizes[128] =
		{ (size_t)NA_INTEGER, (size_t)NA_INTEGER, (size_t)NA_INTEGER };
	clGetDeviceInfo(gpu_device, CL_DEVICE_MAX_WORK_ITEM_SIZES,
		sizeof(gl_max_work_item_sizes), gl_max_work_item_sizes, NULL);

	SEXP rv_ans = PROTECT(NEW_LIST(10));
	SET_ELEMENT(rv_ans, 0, Rf_ScalarReal(gl_mem_sz));
	SET_ELEMENT(rv_ans, 1, Rf_ScalarReal(gl_mem_alloc_sz));
	SET_ELEMENT(rv_ans, 2, Rf_ScalarInteger(gl_n_unit));
	SET_ELEMENT(rv_ans, 3, Rf_ScalarInteger(gl_max_worksize));
	SET_ELEMENT(rv_ans, 4, Rf_ScalarInteger(gl_max_workdim));

	SEXP p_wis = NEW_INTEGER(3);
	INTEGER(p_wis)[0] = gl_max_work_item_sizes[0];
	INTEGER(p_wis)[1] = gl_max_work_item_sizes[1];
	INTEGER(p_wis)[2] = gl_max_work_item_sizes[2];
	SET_ELEMENT(rv_ans, 5, p_wis);

	SET_ELEMENT(rv_ans, 6, Rf_ScalarInteger(gl_local_mem_sz));
	SET_ELEMENT(rv_ans, 7, Rf_ScalarInteger(gl_addr_bits));
	UNPROTECT(1);

	return rv_ans;
	#undef GET_DEV_INFO
}


/// select an OpenCL device
SEXP ocl_select_dev(SEXP dev_idx)
{
	static const char *fc_create_ctx = "clCreateContext";
	static const char *fc_create_queue = "clCreateCommandQueue";
	static const char *err_info = "Failed to select and initialize the device";

	const int idx = Rf_asInteger(dev_idx);
	check_dev_idx(idx);
	cl_device_id dev = gpu_device = gpu_dev_list[idx-1];

	// release the previous device
	ocl_release_dev();

	// start initializing

	cl_int err;
	cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    if (!ctx || err!=CL_SUCCESS)
		Rf_error(gpu_err_msg(err, err_info, fc_create_ctx));
	gpu_context = ctx;

	cl_command_queue queue = clCreateCommandQueue(ctx, dev,
		CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
	if (!queue && err==CL_INVALID_VALUE)
		queue = clCreateCommandQueue(ctx, dev, 0, &err); // not support out-of-order flag
	if (!queue || err!=CL_SUCCESS)
		Rf_error(gpu_err_msg(err, err_info, fc_create_queue));
	gpu_command_queue = queue;

	// initialize mem_rare_freq_f32 and mem_rare_freq_f64
	const double MIN_RARE_FREQ = 1e-5;  // the minimum rare frequency to store haplotypes (defined in HIBAG)
	const int n = 2 * HLA_LIB::HIBAG_MAXNUM_SNP_IN_CLASSIFIER + 1;
	float  mfreq32[n];
	double mfreq64[n];
	for (int i=0; i < n; i++)
		mfreq32[i] = mfreq64[i] = exp(i * log(MIN_RARE_FREQ));
	GPU_CREATE_MEM(mem_rare_freq_f32, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(mfreq32), mfreq32);
	GPU_CREATE_MEM(mem_rare_freq_f64, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(mfreq64), mfreq64);

	// return
	return R_NilValue;
}


/// release the previous device
SEXP ocl_release_dev()
{
	// release kernels
	const int kl_num = 7;
	static cl_kernel *kl_lst[kl_num] = {
		&gpu_kl_clear_mem, &gpu_kl_build_calc_prob, &gpu_kl_build_calc_oob,
		&gpu_kl_build_calc_ib, &gpu_kl_pred_calc, &gpu_kl_pred_sumprob,
		&gpu_kl_pred_addprob };
	for (int i=0; i < kl_num; i++)
	{
		if (*kl_lst[i])
		{
			clReleaseKernel(*kl_lst[i]);
			*kl_lst[i] = NULL;
		}
	}

	// release command queue
	if (gpu_command_queue)
	{
		clReleaseCommandQueue(gpu_command_queue);
		gpu_command_queue = NULL;
	}
	// release context
	if (gpu_context)
	{
		clReleaseContext(gpu_context);
		gpu_context = NULL;
	}

	return R_NilValue;
}


/// attempt to build a kernel
SEXP ocl_set_kl_attempt(SEXP name, SEXP code)
{
	const char *nm = CHAR(STRING_ELT(name, 0));
	cl_kernel kl = build_kernel(code, nm);
	clReleaseKernel(kl);
	return Rf_ScalarLogical(TRUE);
}


/// set kernel for clear memory
SEXP ocl_set_kl_clearmem(SEXP code)
{
	gpu_kl_clear_mem = build_kernel(code, kl_nm_clear_mem);
	int ws = get_kernel_param(gpu_device, gpu_kl_clear_mem,
		CL_KERNEL_WORK_GROUP_SIZE);
	int mt = get_kernel_param(gpu_device, gpu_kl_clear_mem,
		CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE);
	SEXP rv_ans = PROTECT(NEW_INTEGER(2));
	INTEGER(rv_ans)[0] = ws;
	INTEGER(rv_ans)[1] = mt;
	UNPROTECT(1);
	return rv_ans;
}


/// set kernels for building the model
SEXP ocl_set_kl_build(SEXP f64, SEXP f64_build, SEXP code_prob, SEXP code_oob,
	SEXP code_ib)
{
	gpu_f64_flag = (Rf_asLogical(f64) == TRUE);
	gpu_f64_build_flag = (Rf_asLogical(f64_build) == TRUE);
	gpu_kl_build_calc_prob = build_kernel(code_prob, kl_nm_build_calc_prob);
	gpu_kl_build_calc_oob  = build_kernel(code_oob, kl_nm_build_calc_oob);
	gpu_kl_build_calc_ib   = build_kernel(code_ib, kl_nm_build_calc_ib);
	return R_NilValue;
}


/// set kernels for prediction
SEXP ocl_set_kl_predict(SEXP f64_pred, SEXP code_calc, SEXP code_sum, SEXP code_add)
{
	gpu_f64_pred_flag = (Rf_asLogical(f64_pred) == TRUE);
	gpu_kl_pred_calc = build_kernel(code_calc, kl_nm_pred_calc_prob);
	gpu_kl_pred_sumprob = build_kernel(code_sum, kl_nm_pred_calc_sumprob);
	gpu_kl_pred_addprob = build_kernel(code_add, kl_nm_pred_calc_addprob);
	return R_NilValue;
}


} // extern "C"
