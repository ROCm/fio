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
#include <sys/time.h>
#include <sys/resource.h>
#include <cufile.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pthread.h>

#include "../fio.h"
#include "../lib/pow2.h"
#include "../optgroup.h"
#include "../lib/memalign.h"

#define ALIGNED_4KB(v) (((v) & 0x0fff) == 0)

#define LOGGED_BUFLEN_NOT_ALIGNED     0x01
#define LOGGED_GPU_OFFSET_NOT_ALIGNED 0x02
#define GPU_ID_SEP ":"

enum {
	IO_DIRECT    = 1,
	IO_POSIX     = 2
};

enum {
	MEMCPY_DIRECTION_H2D = 1,
	MEMCPY_DIRECTION_D2H = 2
};

struct gpuaccel_backend {
	const char *name;
	int (*driver_open)(void);
	void (*driver_close)(void);

	int (*set_device)(int gpu_id);
	int (*malloc)(void **mem, size_t size);
	int (*free)(void *mem);
	int (*memset)(void *mem, int value, size_t size);
	int (*memcpy)(void *dst, const void *src, size_t size, int direction);
	int (*stream_sync)(void);

	int (*file_handle_register)(int fd, void **handle);
	void (*file_handle_deregister)(void *handle);

	int (*buf_register)(void *mem, size_t size);
	void (*buf_deregister)(void *mem);

	ssize_t (*read)(void *handle, void *mem, size_t size,
			      unsigned long long file_offset, size_t mem_offset);
	ssize_t (*write)(void *handle, const void *mem, size_t size,
			       unsigned long long file_offset, size_t mem_offset);

	const char *(*op_error_string)(int error_code);
};

struct gpuaccel_options {
	struct thread_data *td;
	char               *gpu_ids;       /* colon-separated list of GPU ids,
					      one per job */
	void               *gpu_mem_ptr;    /* GPU memory */
	void               *junk_buf;      /* buffer to simulate cudaMemcpy with
					      posix I/O write */
	int                 my_gpu_id;     /* GPU id to use for this job */
	unsigned int        io_mode;       /* Type of I/O to use */
	size_t              total_mem;     /* size for gpu_mem_ptr and junk_buf */
	int                 logged;        /* bitmask of log messages that have
					      been output, prevent flood */
	const struct gpuaccel_backend *backend; /* GPU accelerator backend function table */
};

struct fio_libcufile_data {
	CUfileDescr_t  cf_descr;
	CUfileHandle_t cf_handle;
};

static struct fio_option options[] = {
	{
		.name	  = "gpu_dev_ids",
		.lname	  = "libcufile engine gpu dev ids",
		.type	  = FIO_OPT_STR_STORE,
		.off1	  = offsetof(struct gpuaccel_options, gpu_ids),
		.help	  = "GPU IDs, one per subjob, separated by " GPU_ID_SEP,
		.category = FIO_OPT_C_ENGINE,
		.group	  = FIO_OPT_G_LIBCUFILE,
	},
	{
		.name	  = "cuda_io",
		.lname	  = "libcufile io mode",
		.type	  = FIO_OPT_STR,
		.off1	  = offsetof(struct gpuaccel_options, io_mode),
		.help	  = "Type of I/O to use with CUDA",
		.def      = "cufile",
		.posval   = {
			    { .ival = "cufile",
			      .oval = IO_DIRECT,
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
static int gpuaccel_initialized = 0;
static pthread_mutex_t running_lock = PTHREAD_MUTEX_INITIALIZER;

#define check_cudaruntimecall(fn, rc)                                               \
	do {                                                                        \
		cudaError_t res = fn;                                               \
		if (res != cudaSuccess) {                                           \
			const char *str = cudaGetErrorName(res);                    \
			log_err("cuda runtime api call failed %s:%d : err=%d:%s\n", \
				#fn, __LINE__, res, str);                           \
			rc = -1;                                                    \
		} else                                                              \
			rc = 0;                                                     \
	} while(0)

static const char *fio_libcufile_get_cuda_error(CUfileError_t st)
{
	if (IS_CUFILE_ERR(st.err))
		return cufileop_status_error(st.err);
	return "unknown";
}

static int libcufile_driver_open(void)
{
	CUfileError_t status;

	status = cuFileDriverOpen();
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
	int rc;

	check_cudaruntimecall(cudaSetDevice(gpu_id), rc);
	return rc;
}

static int libcufile_malloc(void **mem, size_t size)
{
	int rc;

	check_cudaruntimecall(cudaMalloc(mem, size), rc);
	return rc;
}

static int libcufile_free(void *mem)
{
	int rc;

	check_cudaruntimecall(cudaFree(mem), rc);
	return rc;
}

static int libcufile_memset(void *mem, int value, size_t size)
{
	int rc;

	check_cudaruntimecall(cudaMemset(mem, value, size), rc);
	return rc;
}

static int libcufile_memcpy(void *dst, const void *src, size_t size, int direction)
{
	int rc;
	enum cudaMemcpyKind kind;

	switch (direction) {
	case MEMCPY_DIRECTION_H2D:
		kind = cudaMemcpyHostToDevice;
		break;
	case MEMCPY_DIRECTION_D2H:
		kind = cudaMemcpyDeviceToHost;
		break;
	default:
		return -1;
	}

	check_cudaruntimecall(cudaMemcpy(dst, src, size, kind), rc);
	return rc;
}

static int libcufile_stream_sync(void)
{
	int rc;

	check_cudaruntimecall(cudaStreamSynchronize(NULL), rc);
	return rc;
}

static int libcufile_file_handle_register(int fd, void **handle)
{
	struct fio_libcufile_data *fcd;
	CUfileError_t status;

	fcd = calloc(1, sizeof(*fcd));
	if (fcd == NULL)
		return ENOMEM;

	fcd->cf_descr.handle.fd = fd;
	fcd->cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
	status = cuFileHandleRegister(&fcd->cf_handle, &fcd->cf_descr);
	if (status.err != CU_FILE_SUCCESS) {
		log_err("cufile register: err=%d:%s\n", status.err,
			fio_libcufile_get_cuda_error(status));
		free(fcd);
		return EINVAL;
	}

	*handle = fcd;
	return 0;
}

static void libcufile_file_handle_deregister(void *handle)
{
	struct fio_libcufile_data *fcd = handle;

	cuFileHandleDeregister(fcd->cf_handle);
	free(fcd);
}

static int libcufile_buf_register(void *mem, size_t size)
{
	CUfileError_t status;

	status = cuFileBufRegister(mem, size, 0);
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

static ssize_t libcufile_read(void *handle, void *mem, size_t size, unsigned long long offset, size_t mem_offset)
{
	struct fio_libcufile_data *fcd = handle;
	return cuFileRead(fcd->cf_handle, mem, size, offset, mem_offset);
}

static ssize_t libcufile_write(void *handle, const void *mem, size_t size, unsigned long long offset, size_t mem_offset)
{
	struct fio_libcufile_data *fcd = handle;
	return cuFileWrite(fcd->cf_handle, mem, size, offset, mem_offset);
}

static const char *libcufile_op_error_string(int error_code)
{
	return cufileop_status_error(error_code);
}

static const struct gpuaccel_backend libcufile_backend = {
	.name = "cufile",
	.driver_open = libcufile_driver_open,
	.driver_close = libcufile_driver_close,
	.set_device = libcufile_set_device,
	.malloc = libcufile_malloc,
	.free = libcufile_free,
	.memset = libcufile_memset,
	.memcpy = libcufile_memcpy,
	.stream_sync = libcufile_stream_sync,
	.file_handle_register = libcufile_file_handle_register,
	.file_handle_deregister = libcufile_file_handle_deregister,
	.buf_register = libcufile_buf_register,
	.buf_deregister = libcufile_buf_deregister,
	.read = libcufile_read,
	.write = libcufile_write,
	.op_error_string = libcufile_op_error_string
};

/*
 * Assign GPU to subjob roundrobin, similar to how multiple
 * entries in 'directory' are handled by fio.
 */
static int fio_gpuaccel_find_gpu_id(struct thread_data *td)
{
	struct gpuaccel_options *o = td->eo;
	int gpu_id = 0;

	if (o->gpu_ids != NULL) {
		char *gpu_ids, *pos, *cur;
		int i, id_count, gpu_idx;

		for (id_count = 0, cur = o->gpu_ids; cur != NULL; id_count++) {
			cur = strchr(cur, GPU_ID_SEP[0]);
			if (cur != NULL)
				cur++;
		}

		gpu_idx = td->subjob_number % id_count;

		pos = gpu_ids = strdup(o->gpu_ids);
		if (gpu_ids == NULL) {
			log_err("strdup(gpu_ids): err=%d\n", errno);
			return -1;
		}

		i = 0;
		while (pos != NULL && i <= gpu_idx) {
			i++;
			cur = strsep(&pos, GPU_ID_SEP);
		}

		if (cur)
			gpu_id = atoi(cur);

		free(gpu_ids);
	}

	return gpu_id;
}

static int fio_gpuaccel_init(struct thread_data *td)
{
	struct gpuaccel_options *o = td->eo;
	const struct gpuaccel_backend *be = o->backend;
	int initialized;

	pthread_mutex_lock(&running_lock);
	if (running == 0) {
		assert(gpuaccel_initialized == 0);
		if (o->io_mode == IO_DIRECT) {
			/* only open the driver if this is the first worker thread */
			if (be->driver_open() != 0)
				log_err("%s driver_open failed\n", be->name);
			else
				gpuaccel_initialized = 1;
		}
	}
	running++;
	initialized = gpuaccel_initialized;
	pthread_mutex_unlock(&running_lock);

	if (o->io_mode == IO_DIRECT && !initialized)
		return 1;

	o->my_gpu_id = fio_gpuaccel_find_gpu_id(td);
	if (o->my_gpu_id < 0)
		return 1;

	dprint(FD_MEM, "Subjob %d uses GPU %d\n", td->subjob_number, o->my_gpu_id);
	if (be->set_device(o->my_gpu_id) != 0)
		return 1;

	return 0;
}

static inline int fio_gpuaccel_pre_write(struct thread_data *td,
					  struct gpuaccel_options *o,
					  struct io_u *io_u,
					  size_t gpu_offset)
{
	int rc = 0;
	const struct gpuaccel_backend *be = o->backend;

	if (o->io_mode == IO_DIRECT) {
		if (td->o.verify) {
			/*
			  Data is being verified, copy the io_u buffer to GPU memory.
			  This isn't done in the non-verify case because the data would
			  already be in GPU memory in a normal direct io application.
			*/
			rc = be->memcpy(((char*) o->gpu_mem_ptr) + gpu_offset,
					io_u->xfer_buf,
					io_u->xfer_buflen, MEMCPY_DIRECTION_H2D);
			if (rc != 0) {
				log_err("DDIR_WRITE %s memcpy H2D failed\n", be->name);
				io_u->error = EIO;
			}
		}
	} else if (o->io_mode == IO_POSIX) {

		/*
		  POSIX I/O is being used, the data has to be copied out of the
		  GPU into a CPU buffer. GPU memory doesn't contain the actual
		  data to write, copy the data to the junk buffer. The purpose
		  of this is to add the overhead of memcpy() that would be
		  present in a POSIX I/O GPU application.
		*/
		rc = be->memcpy(o->junk_buf + gpu_offset,
				((char*) o->gpu_mem_ptr) + gpu_offset,
				io_u->xfer_buflen, MEMCPY_DIRECTION_D2H);
		if (rc != 0) {
			log_err("DDIR_WRITE %s memcpy D2H failed\n", be->name);
			io_u->error = EIO;
		}
	} else {
		log_err("Illegal %s IO type: %d\n", be->name, o->io_mode);
		assert(0);
		rc = EINVAL;
	}

	return rc;
}

static inline int fio_gpuaccel_post_read(struct thread_data *td,
					  struct gpuaccel_options *o,
					  struct io_u *io_u,
					  size_t gpu_offset)
{
	int rc = 0;
	const struct gpuaccel_backend *be = o->backend;

	if (o->io_mode == IO_DIRECT) {
		if (td->o.verify) {
			/* Copy GPU memory to CPU buffer for verify */
			rc = be->memcpy(io_u->xfer_buf,
							 ((char*) o->gpu_mem_ptr) + gpu_offset,
							 io_u->xfer_buflen,
							 MEMCPY_DIRECTION_D2H);
			if (rc != 0) {
				log_err("DDIR_READ %s memcpy D2H failed\n", be->name);
				io_u->error = EIO;
			}
		}
	} else if (o->io_mode == IO_POSIX) {
		/* POSIX I/O read, copy the CPU buffer to GPU memory */
		rc = be->memcpy(((char*) o->gpu_mem_ptr) + gpu_offset,
						 io_u->xfer_buf,
						 io_u->xfer_buflen,
						 MEMCPY_DIRECTION_H2D);
		if (rc != 0) {
			log_err("DDIR_READ %s memcpy H2D failed\n", be->name);
			io_u->error = EIO;
		}
	} else {
		log_err("Illegal %s IO type: %d\n", be->name, o->io_mode);
		assert(0);
		rc = EINVAL;
	}

	return rc;
}

static enum fio_q_status fio_gpuaccel_queue(struct thread_data *td,
					     struct io_u *io_u)
{
	struct gpuaccel_options *o = td->eo;
	const struct gpuaccel_backend *be = o->backend;
	void *file_handle = FILE_ENG_DATA(io_u->file);
	unsigned long long io_offset;
	ssize_t sz;
	ssize_t remaining;
	size_t xfered;
	size_t gpu_offset;
	int rc;

	if (o->io_mode == IO_DIRECT && file_handle == NULL) {
		io_u->error = EINVAL;
		td_verror(td, EINVAL, "xfer");
		return FIO_Q_COMPLETED;
	}

	fio_ro_check(td, io_u);

	switch(io_u->ddir) {
	case DDIR_SYNC:
		rc = fsync(io_u->file->fd);
		if (rc != 0) {
			io_u->error = errno;
			log_err("fsync: err=%d\n", errno);
		}
		break;

	case DDIR_DATASYNC:
		rc = fdatasync(io_u->file->fd);
		if (rc != 0) {
			io_u->error = errno;
			log_err("fdatasync: err=%d\n", errno);
		}
		break;

	case DDIR_READ:
	case DDIR_WRITE:
		/*
		  There may be a better way to calculate gpu_offset. The intent is
		  that gpu_offset equals the the difference between io_u->xfer_buf and
		  the page-aligned base address for io_u buffers.
		*/
		gpu_offset = io_u->index * io_u->xfer_buflen;
		io_offset = io_u->offset;
		remaining = io_u->xfer_buflen;

		xfered = 0;
		sz = 0;

		assert(gpu_offset + io_u->xfer_buflen <= o->total_mem);

		if (o->io_mode == IO_DIRECT) {
			if (!(ALIGNED_4KB(io_u->xfer_buflen) ||
			      (o->logged & LOGGED_BUFLEN_NOT_ALIGNED))) {
				log_err("buflen not 4KB-aligned: %llu\n", io_u->xfer_buflen);
				o->logged |= LOGGED_BUFLEN_NOT_ALIGNED;
			}

			if (!(ALIGNED_4KB(gpu_offset) ||
			      (o->logged & LOGGED_GPU_OFFSET_NOT_ALIGNED))) {
				log_err("gpu_offset not 4KB-aligned: %lu\n", gpu_offset);
				o->logged |= LOGGED_GPU_OFFSET_NOT_ALIGNED;
			}
		}

		if (io_u->ddir == DDIR_WRITE)
			rc = fio_gpuaccel_pre_write(td, o, io_u, gpu_offset);

		if (io_u->error != 0)
			break;

		while (remaining > 0) {
			assert(gpu_offset + xfered <= o->total_mem);
			if (io_u->ddir == DDIR_READ) {
				if (o->io_mode == IO_DIRECT) {
					sz = be->read(file_handle, o->gpu_mem_ptr, remaining,
							io_offset + xfered, gpu_offset + xfered);
					if (sz == -1) {
						io_u->error = errno;
						log_err("%s Read: err=%d\n", be->name, errno);
					} else if (sz < 0) {
						io_u->error = EIO;
						log_err("%s Read: err=%ld:%s\n", be->name, sz,
							be->op_error_string(-sz));
					}
				} else if (o->io_mode == IO_POSIX) {
					sz = pread(io_u->file->fd, ((char*) io_u->xfer_buf) + xfered,
						   remaining, io_offset + xfered);
					if (sz < 0) {
						io_u->error = errno;
						log_err("pread: err=%d\n", errno);
					}
				} else {
					log_err("Illegal CUDA IO type: %d\n", o->io_mode);
					io_u->error = -1;
					assert(0);
				}
			} else if (io_u->ddir == DDIR_WRITE) {
				if (o->io_mode == IO_DIRECT) {
					sz = be->write(file_handle, o->gpu_mem_ptr, remaining,
							 io_offset + xfered, gpu_offset + xfered);
					if (sz == -1) {
						io_u->error = errno;
						log_err("%s Write: err=%d\n", be->name, errno);
					} else if (sz < 0) {
						io_u->error = EIO;
						log_err("%s Write: err=%ld:%s\n", be->name, sz,
							be->op_error_string(-sz));
					}
				} else if (o->io_mode == IO_POSIX) {
					sz = pwrite(io_u->file->fd,
						    ((char*) io_u->xfer_buf) + xfered,
						    remaining, io_offset + xfered);
					if (sz < 0) {
						io_u->error = errno;
						log_err("pwrite: err=%d\n", errno);
					}
				} else {
					log_err("Illegal %s IO type: %d\n", be->name, o->io_mode);
					io_u->error = -1;
					assert(0);
				}
			} else {
				log_err("not DDIR_READ or DDIR_WRITE: %d\n", io_u->ddir);
				io_u->error = -1;
				assert(0);
				break;
			}

			if (io_u->error != 0)
				break;

			remaining -= sz;
			xfered += sz;

			if (remaining != 0)
				log_info("Incomplete %s: %ld bytes remaining\n",
					 io_u->ddir == DDIR_READ? "read" : "write", remaining);
		}

		if (io_u->error != 0)
			break;

		if (io_u->ddir == DDIR_READ)
			rc = fio_gpuaccel_post_read(td, o, io_u, gpu_offset);
		break;

	default:
		io_u->error = EINVAL;
		break;
	}

	if (io_u->error != 0) {
		log_err("IO failed\n");
		td_verror(td, io_u->error, "xfer");
	}

	return FIO_Q_COMPLETED;
}

static int fio_gpuaccel_open_file(struct thread_data *td, struct fio_file *f)
{
	struct gpuaccel_options *o = td->eo;
	const struct gpuaccel_backend *be = o->backend;
	void *handle = NULL;
	int rc;

	rc = generic_open_file(td, f);
	if (rc)
		return rc;

	if (o->io_mode == IO_DIRECT) {
		rc = be->file_handle_register(f->fd, &handle);
		if (rc != 0) {
			goto exit_err;
		}
	}

	FILE_SET_ENG_DATA(f, handle);
	return 0;

exit_err:
	if (handle) {
		free(handle);
		handle = NULL;
	}
	if (f) {
		int rc2 = generic_close_file(td, f);
		if (rc2)
			log_err("generic_close_file: err=%d\n", rc2);
	}
	return rc;
}

static int fio_gpuaccel_close_file(struct thread_data *td, struct fio_file *f)
{
	void *handle = FILE_ENG_DATA(f);
	int rc;
	const struct gpuaccel_backend *be = td->eo->backend;

	if (handle != NULL) {
		be->file_handle_deregister(handle);
		FILE_SET_ENG_DATA(f, NULL);
		free(handle);
	}

	rc = generic_close_file(td, f);

	return rc;
}

static int fio_gpuaccel_iomem_alloc(struct thread_data *td, size_t total_mem)
{
	struct gpuaccel_options *o = td->eo;
	const struct gpuaccel_backend *be = o->backend;
	int rc;

	o->total_mem = total_mem;
	o->logged = 0;
	o->gpu_mem_ptr = NULL;
	o->junk_buf = NULL;
	td->orig_buffer = calloc(1, total_mem);
	if (!td->orig_buffer) {
		log_err("orig_buffer calloc failed: err=%d\n", errno);
		goto exit_error;
	}

	if (o->io_mode == IO_POSIX) {
		o->junk_buf = calloc(1, total_mem);
		if (o->junk_buf == NULL) {
			log_err("junk_buf calloc failed: err=%d\n", errno);
			goto exit_error;
		}
	}

	dprint(FD_MEM, "Alloc %zu for GPU %d\n", total_mem, o->my_gpu_id);
	rc = be->malloc(&o->gpu_mem_ptr, total_mem);
	if (rc != 0)
		goto exit_error;
	rc = be->memset(o->gpu_mem_ptr, 0xab, total_mem);
	if (rc != 0)
		goto exit_error;

	if (o->io_mode == IO_DIRECT) {
		rc = be->buf_register(o->gpu_mem_ptr, total_mem);
		if (rc != 0)
			goto exit_error;
	}

	return 0;

exit_error:
	if (td->orig_buffer) {
		free(td->orig_buffer);
		td->orig_buffer = NULL;
	}
	if (o->junk_buf) {
		free(o->junk_buf);
		o->junk_buf = NULL;
	}
	if (o->gpu_mem_ptr) {
		be->free(o->gpu_mem_ptr);
		o->gpu_mem_ptr = NULL;
	}
	return 1;
}

static void fio_gpuaccel_iomem_free(struct thread_data *td)
{
	struct gpuaccel_options *o = td->eo;
	const struct gpuaccel_backend *be = o->backend;

	if (o->junk_buf) {
		free(o->junk_buf);
		o->junk_buf = NULL;
	}
	if (o->gpu_mem_ptr) {
		if (o->io_mode == IO_DIRECT)
			be->buf_deregister(o->gpu_mem_ptr);
		be->free(o->gpu_mem_ptr);
		o->gpu_mem_ptr = NULL;
	}
	if (td->orig_buffer) {
		free(td->orig_buffer);
		td->orig_buffer = NULL;
	}
}

static void fio_gpuaccel_cleanup(struct thread_data *td)
{
	struct gpuaccel_options *o = td->eo;
	const struct gpuaccel_backend *be = o->backend;

	pthread_mutex_lock(&running_lock);
	running--;
	assert(running >= 0);
	if (running == 0) {
		/* only close the driver if initialized and
		   this is the last worker thread */
		if (o->io_mode == IO_DIRECT && gpuaccel_initialized)
			be->driver_close();
		gpuaccel_initialized = 0;
	}
	pthread_mutex_unlock(&running_lock);
}

static int fio_libcufile_init(struct thread_data *td)
{
	struct gpuaccel_options *o = td->eo;
	o->backend = &libcufile_backend;
	return fio_gpuaccel_init(td);
}

FIO_STATIC struct ioengine_ops ioengine = {
	.name                = "libcufile",
	.version             = FIO_IOOPS_VERSION,
	.init                = fio_libcufile_init,
	.queue               = fio_gpuaccel_queue,
	.get_file_size       = generic_get_file_size,
	.open_file           = fio_gpuaccel_open_file,
	.close_file          = fio_gpuaccel_close_file,
	.iomem_alloc         = fio_gpuaccel_iomem_alloc,
	.iomem_free          = fio_gpuaccel_iomem_free,
	.cleanup             = fio_gpuaccel_cleanup,
	.flags               = FIO_SYNCIO,
	.options             = options,
	.option_struct_size  = sizeof(struct gpuaccel_options)
};

void fio_init fio_libcufile_register(void)
{
	register_ioengine(&ioengine);
}

void fio_exit fio_libcufile_unregister(void)
{
	unregister_ioengine(&ioengine);
}
