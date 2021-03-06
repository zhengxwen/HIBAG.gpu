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

#define USE_RINTERNALS 1
#include <Rinternals.h>
#include <R_ext/Rdynload.h>


extern "C"
{
// OpenCL device list



// OpenCL device variables

cl_context gpu_context = NULL;
cl_command_queue gpu_command_queue = NULL;



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



// ===================================================================== //

std::vector<cl_device_id> gpu_dev_list;


SEXP ocl_get_dev_list(SEXP Rverbose)
{
	static const char *fc_get_platform      = "clGetPlatformIDs";
	static const char *fc_get_platform_info = "clGetPlatformInfo";
	static const char *fc_get_device      = "clGetDeviceIDs";
	static const char *fc_get_device_info = "clGetDeviceInfo";
	static const char *err_info = "It failed to get a list of devices";

	// show information or not
	const bool verbose = Rf_asLogical(Rverbose) == TRUE;

	// clear the list
	gpu_dev_list.clear();

	// get a list of platforms
	cl_uint num_platforms = 0;
	cl_int err = clGetPlatformIDs(0, NULL, &num_platforms);
	if (err != CL_SUCCESS)
		error(gpu_err_msg(err, err_info, fc_get_platform));
	cl_platform_id *platforms =
		(cl_platform_id*)malloc(num_platforms*sizeof(cl_platform_id));
	err = clGetPlatformIDs(num_platforms, platforms, NULL);
	if (err != CL_SUCCESS)
	{
		free(platforms);
		error(gpu_err_msg(err, err_info, fc_get_platform));
	}

	// get a list of devices for each platform
	for (cl_uint i=0; i < num_platforms; i++)
	{
		cl_platform_id platform = platforms[i];
		if (verbose)
		{
			// get platform information
			char ss[1024];
			// platform name
			err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(ss), &ss, NULL);
			if (err != CL_SUCCESS)
			{
				free(platforms);
				error(gpu_err_msg(err, err_info, fc_get_platform_info));
			}
			Rprintf("%s, ", ss);
			// platform version
			err = clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(ss), &ss, NULL);
			if (err != CL_SUCCESS)
			{
				free(platforms);
				error(gpu_err_msg(err, err_info, fc_get_platform_info));
			}
			Rprintf("%s:\n", ss);
		}
		// get devices using this platform
		cl_uint num_dev = 0;
		err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_dev);
		if (err!=CL_SUCCESS && err!=CL_DEVICE_NOT_FOUND)
		{
			free(platforms);
			error(gpu_err_msg(err, err_info, fc_get_device));
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
			{
				gpu_dev_list.push_back(dev[j]);
				if (verbose)
				{
					Rprintf("    Dev #%d: ", (int)gpu_dev_list.size());
					// get device information
					char vendor[1024], name[1024], version[1024];
					err = clGetDeviceInfo(dev[j], CL_DEVICE_VENDOR, sizeof(vendor), &vendor, NULL);
					if (err != CL_SUCCESS)
					{
						free(dev); free(platforms);
						error(gpu_err_msg(err, err_info, fc_get_device_info));
					}
					err = clGetDeviceInfo(dev[j], CL_DEVICE_NAME, sizeof(name), &name, NULL);
					if (err != CL_SUCCESS)
					{
						free(dev); free(platforms);
						error(gpu_err_msg(err, err_info, fc_get_device_info));
					}
					err = clGetDeviceInfo(dev[j], CL_DEVICE_VERSION, sizeof(version), &version, NULL);
					if (err != CL_SUCCESS)
					{
						free(dev); free(platforms);
						error(gpu_err_msg(err, err_info, fc_get_device_info));
					}
					Rprintf("%s, %s, %s\n", vendor, name, version);
				}
			}
			free(dev);
		}
	}

	// finally
	free(platforms);
	return R_NilValue;
}


/*
void R_init_HIBAG_gpu(DllInfo *info)
{
	Rprintf("sjsjs\n");
}

void R_unload_HIBAG_gpu(DllInfo *info)
{
	Rprintf("osksks\n");
}
*/

} // extern "C"
