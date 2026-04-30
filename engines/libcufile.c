/*
 * Copyright (c)2020 System Fabric Works, Inc. All Rights Reserved.
 * mailto:info@systemfabricworks.com
 *
 * License: GPLv2, see COPYING.
 *
 * libcufile engine
 *
 * fio I/O engine using the NVIDIA cuFile API.
 *
 */

#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <cufile.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pthread.h>

#include "../fio.h"
#include "../optgroup.h"
#include "gpuaccel.h"

enum {
	IO_CUFILE = GPUACCEL_IO_DIRECT_MODE,
	IO_POSIX = GPUACCEL_IO_POSIX_MODE,
};

struct libcufile_options {
	struct gpuaccel_options gopts;
	unsigned int cuda_io;
};

struct libcufile_file_handle {
	CUfileHandle_t cf_handle;
};

static struct fio_option options[] = {
	{
		.name	  = "gpu_dev_ids",
		.lname	  = "libcufile engine gpu dev ids",
		.type	  = FIO_OPT_STR_STORE,
		.off1	  = offsetof(struct libcufile_options, gopts.gpu_ids),
		.help	  = "GPU IDs, one per subjob, separated by " GPUACCEL_GPU_ID_SEP,
		.category = FIO_OPT_C_ENGINE,
		.group	  = FIO_OPT_G_LIBCUFILE,
	},
	{
		.name	  = "cuda_io",
		.lname	  = "libcufile cuda io",
		.type	  = FIO_OPT_STR,
		.off1	  = offsetof(struct libcufile_options, cuda_io),
		.help	  = "Type of I/O to use with CUDA",
		.def	  = "cufile",
		.posval   = {
			    { .ival = "cufile",
			      .oval = IO_CUFILE,
			      .help = "libcufile nvidia-fs"
			    },
			    { .ival = "posix",
			      .oval = IO_POSIX,
			      .help = "POSIX I/O"
			    }
		},
		.category = FIO_OPT_C_ENGINE,
		.group	  = FIO_OPT_G_LIBCUFILE,
	},
	{
		.name	 = NULL,
	},
};

static int running = 0;
static int cufile_initialized = 0;
static pthread_mutex_t running_lock = PTHREAD_MUTEX_INITIALIZER;

static const char *fio_libcufile_get_cuda_error(CUfileError_t st)
{
	if (IS_CUFILE_ERR(st.err))
		return cufileop_status_error(st.err);
	return "unknown";
}

static int libcufile_check_cuda(cudaError_t res, const char *fn)
{
	if (res != cudaSuccess) {
		const char *str = cudaGetErrorName(res);
		log_err("cuda runtime api call failed %s : err=%d:%s\n", fn, res, str);
		return -1;
	}

	return 0;
}

static int libcufile_driver_open(void)
{
	CUfileError_t status = cuFileDriverOpen();

	if (status.err != CU_FILE_SUCCESS) {
		log_err("cuFileDriverOpen: err=%d:%s\n", status.err,
			fio_libcufile_get_cuda_error(status));
		return -1;
	}

	return 0;
}

static void libcufile_driver_close(void)
{
	cuFileDriverClose();
}

static int libcufile_set_device(int gpu_id)
{
	return libcufile_check_cuda(cudaSetDevice(gpu_id), "cudaSetDevice");
}

static int libcufile_mem_alloc(void **mem, size_t size)
{
	return libcufile_check_cuda(cudaMalloc(mem, size), "cudaMalloc");
}

static int libcufile_mem_free(void *mem)
{
	return libcufile_check_cuda(cudaFree(mem), "cudaFree");
}

static int libcufile_mem_set(void *mem, int value, size_t size)
{
	return libcufile_check_cuda(cudaMemset(mem, value, size), "cudaMemset");
}

static int libcufile_mem_copy(void *dst, const void *src, size_t size, int direction)
{
	cudaMemcpyKind kind;

	switch (direction) {
	case GPUACCEL_MEMCPY_H2D:
		kind = cudaMemcpyHostToDevice;
		break;
	case GPUACCEL_MEMCPY_D2H:
		kind = cudaMemcpyDeviceToHost;
		break;
	default:
		return -1;
	}

	return libcufile_check_cuda(cudaMemcpy(dst, src, size, kind), "cudaMemcpy");
}

static int libcufile_stream_sync(void)
{
	return libcufile_check_cuda(cudaStreamSynchronize(NULL),
				    "cudaStreamSynchronize");
}

static int libcufile_file_handle_register(int fd, void **handle)
{
	struct libcufile_file_handle *fch;
	CUfileDescr_t cf_descr = { 0 };
	CUfileError_t status;

	fch = calloc(1, sizeof(*fch));
	if (!fch)
		return ENOMEM;

	cf_descr.handle.fd = fd;
	cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
	status = cuFileHandleRegister(&fch->cf_handle, &cf_descr);
	if (status.err != CU_FILE_SUCCESS) {
		log_err("cufile register: err=%d:%s\n", status.err,
			fio_libcufile_get_cuda_error(status));
		free(fch);
		return EINVAL;
	}

	*handle = fch;
	return 0;
}

static void libcufile_file_handle_deregister(void *handle)
{
	struct libcufile_file_handle *fch = handle;

	cuFileHandleDeregister(fch->cf_handle);
	free(fch);
}

static int libcufile_buf_register(void *mem, size_t size)
{
	CUfileError_t status = cuFileBufRegister(mem, size, 0);

	if (status.err != CU_FILE_SUCCESS) {
		log_err("cuFileBufRegister: err=%d:%s\n", status.err,
			fio_libcufile_get_cuda_error(status));
		return -1;
	}

	return 0;
}

static void libcufile_buf_deregister(void *mem)
{
	cuFileBufDeregister(mem);
}

static ssize_t libcufile_direct_read(void *handle, void *mem, size_t size,
				     unsigned long long file_offset, size_t mem_offset)
{
	struct libcufile_file_handle *fch = handle;

	return cuFileRead(fch->cf_handle, mem, size, file_offset, mem_offset);
}

static ssize_t libcufile_direct_write(void *handle, const void *mem, size_t size,
				      unsigned long long file_offset, size_t mem_offset)
{
	struct libcufile_file_handle *fch = handle;

	return cuFileWrite(fch->cf_handle, mem, size, file_offset, mem_offset);
}

static const char *libcufile_op_error_string(int error_code)
{
	return cufileop_status_error(error_code);
}

static const struct gpuaccel_backend libcufile_backend = {
	.name = "CUDA",
	.direct_io_name = "cuFile",
	.direct_mode = IO_CUFILE,
	.posix_mode = IO_POSIX,
	.running = &running,
	.initialized = &cufile_initialized,
	.running_lock = &running_lock,
	.driver_open = libcufile_driver_open,
	.driver_close = libcufile_driver_close,
	.set_device = libcufile_set_device,
	.mem_alloc = libcufile_mem_alloc,
	.mem_free = libcufile_mem_free,
	.mem_set = libcufile_mem_set,
	.mem_copy = libcufile_mem_copy,
	.stream_sync = libcufile_stream_sync,
	.file_handle_register = libcufile_file_handle_register,
	.file_handle_deregister = libcufile_file_handle_deregister,
	.buf_register = libcufile_buf_register,
	.buf_deregister = libcufile_buf_deregister,
	.direct_read = libcufile_direct_read,
	.direct_write = libcufile_direct_write,
	.op_error_string = libcufile_op_error_string,
	.sync_after_posix_write_copy = 0,
	.sync_after_verify_read_copy = 0,
	.sync_after_memset = 0,
};

static int fio_libcufile_init(struct thread_data *td)
{
	struct libcufile_options *o = td->eo;

	o->gopts.io_mode = o->cuda_io;
	o->gopts.backend = &libcufile_backend;

	return fio_gpuaccel_init(td);
}

FIO_STATIC struct ioengine_ops ioengine = {
	.name               = "libcufile",
	.version            = FIO_IOOPS_VERSION,
	.init               = fio_libcufile_init,
	.queue              = fio_gpuaccel_queue,
	.get_file_size      = generic_get_file_size,
	.open_file          = fio_gpuaccel_open_file,
	.close_file         = fio_gpuaccel_close_file,
	.iomem_alloc        = fio_gpuaccel_iomem_alloc,
	.iomem_free         = fio_gpuaccel_iomem_free,
	.cleanup            = fio_gpuaccel_cleanup,
	.flags              = FIO_SYNCIO,
	.options            = options,
	.option_struct_size = sizeof(struct libcufile_options),
};

void fio_init fio_libcufile_register(void)
{
	register_ioengine(&ioengine);
}

void fio_exit fio_libcufile_unregister(void)
{
	unregister_ioengine(&ioengine);
}