/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 * mailto: hipfile-maintainer@amd.com
 *
 * License: GPLv2, see COPYING.
 *
 * libhipfile engine
 *
 * FIO gpuaccel engine implementation for AMD ROCm hipfile API.
 */

#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <hipfile.h>
#include <hip/hip_runtime.h>

#include "../fio.h"
#include "../optgroup.h"
#include "gpuaccel.h"

struct libhipfile_file_data {
	hipFileDescr_t hf_descr;
	hipFileHandle_t hf_handle;
};

static struct fio_option options[] = {
	{
		.name	  = "gpu_dev_ids",
		.lname	  = "libhipfile engine gpu dev ids",
		.type	  = FIO_OPT_STR_STORE,
		.off1	  = offsetof(struct gpuaccel_options, gpu_ids),
		.help	  = "GPU IDs, one per subjob, separated by " GPU_ID_SEP,
		.category = FIO_OPT_C_ENGINE,
		.group	  = FIO_OPT_G_LIBHIPFILE,
	},
	{
		.name	  = "rocm_io",
		.lname	  = "libhipfile rocm io",
		.type	  = FIO_OPT_STR,
		.off1	  = offsetof(struct gpuaccel_options, io_mode),
		.help	  = "Type of I/O to use with ROCm",
		.def      = "hipfile",
		.posval   = {
			    { .ival = "hipfile",
			      .oval = IO_DIRECT,
			      .help = "libhipfile"
			    },
			    { .ival = "posix",
			      .oval = IO_POSIX,
			      .help = "POSIX I/O"
			    }
		},
		.category = FIO_OPT_C_ENGINE,
		.group	  = FIO_OPT_G_LIBHIPFILE,
	},
	{
		.name	  = "hipfile_mode",
		.lname	  = "libhipfile submission mode",
		.type	  = FIO_OPT_STR,
		.off1	  = offsetof(struct gpuaccel_options, submit_mode),
		.help	  = "hipfile I/O submission model (rocm_io=hipfile only)",
		.def      = "sync",
		.posval   = {
			    { .ival = "sync",
			      .oval = SUBMIT_SYNC,
			      .help = "synchronous hipFileRead/Write"
			    },
			    { .ival = "batch",
			      .oval = SUBMIT_BATCH,
			      .help = "async hipfile batch API"
			    },
			    { .ival = "stream",
			      .oval = SUBMIT_STREAM,
			      .help = "async hipfile stream API"
			    }
		},
		.category = FIO_OPT_C_ENGINE,
		.group	  = FIO_OPT_G_LIBHIPFILE,
	},
	{
		.name	 = NULL,
	},
};

static const char *fio_libhipfile_get_hip_error(hipFileError_t st)
{
	if (st.err > HIPFILE_BASE_ERR)
		return hipFileGetOpErrorString(st.err);
	return "unknown";
}

static int libhipfile_check_runtime(hipError_t res, const char *fn)
{
	if (res != hipSuccess) {
		const char *str = hipGetErrorName(res);
		log_err("hip runtime api call failed %s : err=%d:%s\n", fn, res, str);
		return -1;
	}

	return 0;
}

static int libhipfile_driver_open(void)
{
	hipFileError_t status = hipFileDriverOpen();

	if (status.err != hipFileSuccess) {
		log_err("hipFileDriverOpen: err=%d:%s\n", status.err,
			fio_libhipfile_get_hip_error(status));
		return -1;
	}

	return 0;
}

static void libhipfile_driver_close(void)
{
	hipFileDriverClose();
}

static int libhipfile_set_device(int gpu_id)
{
	return libhipfile_check_runtime(hipSetDevice(gpu_id), "hipSetDevice");
}

static int libhipfile_malloc(void **mem, size_t size)
{
	return libhipfile_check_runtime(hipMalloc(mem, size), "hipMalloc");
}

static int libhipfile_free(void *mem)
{
	return libhipfile_check_runtime(hipFree(mem), "hipFree");
}

static int libhipfile_memset(void *mem, int value, size_t size)
{
	return libhipfile_check_runtime(hipMemset(mem, value, size), "hipMemset");
}

static int libhipfile_memcpy(void *dst, const void *src, size_t size, int direction)
{
	enum hipMemcpyKind kind;

	switch (direction) {
	case MEMCPY_DIRECTION_H2D:
		kind = hipMemcpyHostToDevice;
		break;
	case MEMCPY_DIRECTION_D2H:
		kind = hipMemcpyDeviceToHost;
		break;
	default:
		return -1;
	}

	return libhipfile_check_runtime(hipMemcpy(dst, src, size, kind), "hipMemcpy");
}

static int libhipfile_stream_sync(void)
{
	return libhipfile_check_runtime(hipStreamSynchronize(NULL),
				       "hipStreamSynchronize");
}

static int libhipfile_file_handle_register(int fd, void **handle)
{
	struct libhipfile_file_data *fhd;
	hipFileError_t status;

	fhd = calloc(1, sizeof(*fhd));
	if (!fhd)
		return ENOMEM;

	fhd->hf_descr.handle.fd = fd;
	fhd->hf_descr.type = hipFileHandleTypeOpaqueFD;
	status = hipFileHandleRegister(&fhd->hf_handle, &fhd->hf_descr);
	if (status.err != hipFileSuccess) {
		log_err("hipFileHandleRegister: err=%d:%s\n", status.err,
			fio_libhipfile_get_hip_error(status));
		free(fhd);
		return EINVAL;
	}

	*handle = fhd;
	return 0;
}

static void libhipfile_file_handle_deregister(void *handle)
{
	struct libhipfile_file_data *fhd = handle;

	hipFileHandleDeregister(fhd->hf_handle);
	free(fhd);
}

static int libhipfile_buf_register(void *mem, size_t size)
{
	hipFileError_t status = hipFileBufRegister(mem, size, 0);

	if (status.err != hipFileSuccess) {
		log_err("hipFileBufRegister: err=%d:%s\n", status.err,
			fio_libhipfile_get_hip_error(status));
		return -1;
	}

	return 0;
}

static void libhipfile_buf_deregister(void *mem)
{
	hipFileBufDeregister(mem);
}

static ssize_t libhipfile_read(void *handle, void *mem, size_t size,
				      unsigned long long file_offset, size_t mem_offset)
{
	struct libhipfile_file_data *fhd = handle;

	return hipFileRead(fhd->hf_handle, mem, size, file_offset, mem_offset);
}

static ssize_t libhipfile_write(void *handle, const void *mem, size_t size,
				       unsigned long long file_offset, size_t mem_offset)
{
	struct libhipfile_file_data *fhd = handle;

	return hipFileWrite(fhd->hf_handle, mem, size, file_offset, mem_offset);
}

static const char *libhipfile_op_error_string(int error_code)
{
	return hipFileGetOpErrorString(error_code);
}

/* ------------------------------------------------------------------ *
 * Batch async submission (hipFileBatchIO*)
 * ------------------------------------------------------------------ */

struct libhipfile_batch_ctx {
	hipFileBatchHandle_t batch;
	hipFileIOParams_t   *params;   /* depth slots, filled by prep */
	hipFileIOEvents_t   *events;   /* depth slots, filled by reap */
	unsigned int         depth;
	unsigned int         staged;   /* params filled since last submit */
};

static int libhipfile_batch_setup(struct thread_data *td, void **ctxp,
				  unsigned int depth)
{
	struct libhipfile_batch_ctx *bc;
	hipFileError_t status;

	bc = calloc(1, sizeof(*bc));
	if (!bc)
		return ENOMEM;

	bc->depth = depth;
	bc->params = calloc(depth, sizeof(*bc->params));
	bc->events = calloc(depth, sizeof(*bc->events));
	if (!bc->params || !bc->events) {
		free(bc->params);
		free(bc->events);
		free(bc);
		return ENOMEM;
	}

	status = hipFileBatchIOSetUp(&bc->batch, depth);
	if (status.err != hipFileSuccess) {
		log_err("hipFileBatchIOSetUp: err=%d:%s\n", status.err,
			fio_libhipfile_get_hip_error(status));
		free(bc->params);
		free(bc->events);
		free(bc);
		return EINVAL;
	}

	*ctxp = bc;
	return 0;
}

static void libhipfile_batch_destroy(struct thread_data *td, void *ctx)
{
	struct libhipfile_batch_ctx *bc = ctx;

	if (!bc)
		return;

	hipFileBatchIODestroy(bc->batch);
	free(bc->params);
	free(bc->events);
	free(bc);
}

static int libhipfile_batch_prep(void *ctx, struct io_u *io_u, void *handle,
				 void *mem, size_t size,
				 unsigned long long file_off, size_t mem_off,
				 int ddir)
{
	struct libhipfile_batch_ctx *bc = ctx;
	struct libhipfile_file_data *fhd = handle;
	hipFileIOParams_t *p;

	if (bc->staged >= bc->depth)
		return -1;

	p = &bc->params[bc->staged++];
	p->mode = hipFileBatch;
	p->fh = fhd->hf_handle;
	p->opcode = (ddir == DDIR_WRITE) ? hipFileBatchWrite : hipFileBatchRead;
	p->u.batch.devPtr_base = mem;
	p->u.batch.devPtr_offset = mem_off;
	p->u.batch.file_offset = file_off;
	p->u.batch.size = size;
	p->cookie = io_u;
	return 0;
}

static int libhipfile_batch_submit(void *ctx, unsigned int nr)
{
	struct libhipfile_batch_ctx *bc = ctx;
	hipFileError_t status;

	status = hipFileBatchIOSubmit(bc->batch, nr, bc->params, 0);
	bc->staged = 0;
	if (status.err != hipFileSuccess) {
		log_err("hipFileBatchIOSubmit: err=%d:%s\n", status.err,
			fio_libhipfile_get_hip_error(status));
		return -1;
	}
	return 0;
}

static int libhipfile_batch_reap(void *ctx, unsigned int min, unsigned int max,
				 struct gpuaccel_io_event *events,
				 const struct timespec *t)
{
	struct libhipfile_batch_ctx *bc = ctx;
	hipFileError_t status;
	struct timespec ts, *tsp = NULL;
	unsigned int nr, i;

	if (max > bc->depth)
		max = bc->depth;
	nr = max;

	if (t) {
		ts = *t;
		tsp = &ts;
	}

	status = hipFileBatchIOGetStatus(bc->batch, min, &nr, bc->events, tsp);
	if (status.err != hipFileSuccess) {
		log_err("hipFileBatchIOGetStatus: err=%d:%s\n", status.err,
			fio_libhipfile_get_hip_error(status));
		return -1;
	}

	for (i = 0; i < nr; i++) {
		events[i].io_u = bc->events[i].cookie;
		if (bc->events[i].status == hipFileComplete) {
			events[i].ret = (ssize_t) bc->events[i].ret;
			events[i].error = 0;
		} else {
			events[i].ret = 0;
			events[i].error = EIO;
			log_err("hipfile batch op status=%d\n",
				bc->events[i].status);
		}
	}

	return nr;
}

static const struct gpuaccel_async_ops libhipfile_batch_ops = {
	.setup   = libhipfile_batch_setup,
	.destroy = libhipfile_batch_destroy,
	.prep    = libhipfile_batch_prep,
	.submit  = libhipfile_batch_submit,
	.reap    = libhipfile_batch_reap,
};

/* ------------------------------------------------------------------ *
 * Stream async submission (hipFileReadAsync/hipFileWriteAsync)
 * ------------------------------------------------------------------ */

enum {
	HF_SLOT_FREE = 0,
	HF_SLOT_STAGED,
	HF_SLOT_INFLIGHT
};

struct libhipfile_stream_slot {
	int             state;
	struct io_u    *io_u;
	hipEvent_t      event;
	hipFileHandle_t fh;
	void           *dev_base;
	/* by-pointer async args must persist until the stream op executes */
	size_t          size;
	hoff_t          file_offset;
	hoff_t          buf_offset;
	ssize_t         bytes;
	int             ddir;
};

struct libhipfile_stream_ctx {
	hipStream_t                    stream;
	struct libhipfile_stream_slot *slots;  /* depth slots */
	unsigned int                   depth;
};

static int libhipfile_stream_setup(struct thread_data *td, void **ctxp,
				   unsigned int depth)
{
	struct libhipfile_stream_ctx *sc;
	hipFileError_t status;
	unsigned int i;

	sc = calloc(1, sizeof(*sc));
	if (!sc)
		return ENOMEM;

	sc->depth = depth;
	sc->slots = calloc(depth, sizeof(*sc->slots));
	if (!sc->slots) {
		free(sc);
		return ENOMEM;
	}

	if (libhipfile_check_runtime(
		    hipStreamCreateWithFlags(&sc->stream, hipStreamNonBlocking),
		    "hipStreamCreateWithFlags") != 0)
		goto err;

	status = hipFileStreamRegister(sc->stream, 0);
	if (status.err != hipFileSuccess) {
		log_err("hipFileStreamRegister: err=%d:%s\n", status.err,
			fio_libhipfile_get_hip_error(status));
		goto err_stream;
	}

	for (i = 0; i < depth; i++) {
		if (libhipfile_check_runtime(hipEventCreate(&sc->slots[i].event),
					     "hipEventCreate") != 0)
			goto err_events;
	}

	*ctxp = sc;
	return 0;

err_events:
	while (i-- > 0)
		hipEventDestroy(sc->slots[i].event);
	hipFileStreamDeregister(sc->stream);
err_stream:
	hipStreamDestroy(sc->stream);
err:
	free(sc->slots);
	free(sc);
	return EINVAL;
}

static void libhipfile_stream_destroy(struct thread_data *td, void *ctx)
{
	struct libhipfile_stream_ctx *sc = ctx;
	unsigned int i;

	if (!sc)
		return;

	for (i = 0; i < sc->depth; i++)
		hipEventDestroy(sc->slots[i].event);
	hipFileStreamDeregister(sc->stream);
	hipStreamDestroy(sc->stream);
	free(sc->slots);
	free(sc);
}

static int libhipfile_stream_prep(void *ctx, struct io_u *io_u, void *handle,
				  void *mem, size_t size,
				  unsigned long long file_off, size_t mem_off,
				  int ddir)
{
	struct libhipfile_stream_ctx *sc = ctx;
	struct libhipfile_file_data *fhd = handle;
	struct libhipfile_stream_slot *slot = NULL;
	unsigned int i;

	for (i = 0; i < sc->depth; i++) {
		if (sc->slots[i].state == HF_SLOT_FREE) {
			slot = &sc->slots[i];
			break;
		}
	}
	if (!slot)
		return -1;

	slot->state = HF_SLOT_STAGED;
	slot->io_u = io_u;
	slot->fh = fhd->hf_handle;
	slot->dev_base = mem;
	slot->size = size;
	slot->file_offset = file_off;
	slot->buf_offset = mem_off;
	slot->bytes = 0;
	slot->ddir = ddir;
	return 0;
}

static int libhipfile_stream_submit(void *ctx, unsigned int nr)
{
	struct libhipfile_stream_ctx *sc = ctx;
	unsigned int i;
	int ret = 0;

	for (i = 0; i < sc->depth; i++) {
		struct libhipfile_stream_slot *slot = &sc->slots[i];
		hipFileError_t status;

		if (slot->state != HF_SLOT_STAGED)
			continue;

		if (slot->ddir == DDIR_WRITE)
			status = hipFileWriteAsync(slot->fh, slot->dev_base,
						   &slot->size, &slot->file_offset,
						   &slot->buf_offset, &slot->bytes,
						   sc->stream);
		else
			status = hipFileReadAsync(slot->fh, slot->dev_base,
						  &slot->size, &slot->file_offset,
						  &slot->buf_offset, &slot->bytes,
						  sc->stream);

		if (status.err != hipFileSuccess) {
			log_err("hipFile%sAsync: err=%d:%s\n",
				slot->ddir == DDIR_WRITE ? "Write" : "Read",
				status.err, fio_libhipfile_get_hip_error(status));
			slot->state = HF_SLOT_FREE;
			ret = -1;
			continue;
		}

		if (libhipfile_check_runtime(hipEventRecord(slot->event, sc->stream),
					     "hipEventRecord") != 0) {
			slot->state = HF_SLOT_FREE;
			ret = -1;
			continue;
		}

		slot->state = HF_SLOT_INFLIGHT;
	}

	(void) nr;
	return ret;
}

static int libhipfile_stream_reap(void *ctx, unsigned int min, unsigned int max,
				  struct gpuaccel_io_event *events,
				  const struct timespec *t)
{
	struct libhipfile_stream_ctx *sc = ctx;
	unsigned int i;
	int n = 0, synced = 0, inflight;

rescan:
	inflight = 0;
	for (i = 0; i < sc->depth && (unsigned int) n < max; i++) {
		struct libhipfile_stream_slot *slot = &sc->slots[i];
		hipError_t qs;

		if (slot->state != HF_SLOT_INFLIGHT)
			continue;

		qs = hipEventQuery(slot->event);
		if (qs == hipErrorNotReady) {
			inflight = 1;
			continue;
		}

		events[n].io_u = slot->io_u;
		if (qs == hipSuccess) {
			events[n].ret = slot->bytes;
			events[n].error = 0;
		} else {
			events[n].ret = 0;
			events[n].error = EIO;
			log_err("hipEventQuery: err=%d:%s\n", qs,
				hipGetErrorName(qs));
		}
		slot->state = HF_SLOT_FREE;
		n++;
	}

	/*
	 * Block rather than busy-spin: if fio wants more than are ready and ops
	 * are still outstanding, drain the stream once and rescan.
	 */
	if ((unsigned int) n < min && inflight && !synced) {
		synced = 1;
		hipStreamSynchronize(sc->stream);
		goto rescan;
	}

	(void) t;
	return n;
}

static const struct gpuaccel_async_ops libhipfile_stream_ops = {
	.setup   = libhipfile_stream_setup,
	.destroy = libhipfile_stream_destroy,
	.prep    = libhipfile_stream_prep,
	.submit  = libhipfile_stream_submit,
	.reap    = libhipfile_stream_reap,
};

static int running = 0;
static int initialized = 0;
static pthread_mutex_t running_lock = PTHREAD_MUTEX_INITIALIZER;

static const struct gpuaccel_backend libhipfile_backend = {
	.name = "hipfile",
	.sync_after_posix_write_copy = 1,
	.sync_after_verify_read_copy = 1,
	.sync_after_memset = 1,
	.running = &running,
	.initialized = &initialized,
	.running_lock = &running_lock,
	.driver_open = libhipfile_driver_open,
	.driver_close = libhipfile_driver_close,
	.set_device = libhipfile_set_device,
	.malloc = libhipfile_malloc,
	.free = libhipfile_free,
	.memset = libhipfile_memset,
	.memcpy = libhipfile_memcpy,
	.stream_sync = libhipfile_stream_sync,
	.file_handle_register = libhipfile_file_handle_register,
	.file_handle_deregister = libhipfile_file_handle_deregister,
	.buf_register = libhipfile_buf_register,
	.buf_deregister = libhipfile_buf_deregister,
	.read = libhipfile_read,
	.write = libhipfile_write,
	.op_error_string = libhipfile_op_error_string,
	.batch = &libhipfile_batch_ops,
	.stream = &libhipfile_stream_ops,
};

static int fio_libhipfile_init(struct thread_data *td)
{
	struct gpuaccel_options *o = td->eo;
	o->backend = &libhipfile_backend;
	return fio_gpuaccel_init(td);
}

FIO_STATIC struct ioengine_ops ioengine = {
	.name               = "libhipfile",
	.version            = FIO_IOOPS_VERSION,
	.init               = fio_libhipfile_init,
	.queue              = fio_gpuaccel_queue_async,
	.commit             = fio_gpuaccel_commit,
	.getevents          = fio_gpuaccel_getevents,
	.event              = fio_gpuaccel_event,
	.get_file_size      = generic_get_file_size,
	.open_file          = fio_gpuaccel_open_file,
	.close_file         = fio_gpuaccel_close_file,
	.iomem_alloc        = fio_gpuaccel_iomem_alloc,
	.iomem_free         = fio_gpuaccel_iomem_free,
	.cleanup            = fio_gpuaccel_cleanup,
	.options            = options,
	.option_struct_size = sizeof(struct gpuaccel_options),
};

void fio_init fio_libhipfile_register(void)
{
	register_ioengine(&ioengine);
}

void fio_exit fio_libhipfile_unregister(void)
{
	unregister_ioengine(&ioengine);
}
