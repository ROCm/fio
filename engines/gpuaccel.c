/*
 * License: GPLv2, see COPYING.
 *
 * gpuaccel engine
 *
 * Abstract engine for GPU-accelerated I/O engines. See libcufile.c for
 * an example implementation.
 */


#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

#include "../fio.h"
#include "gpuaccel.h"

#define ALIGNED_4KB(v) (((v) & 0x0fff) == 0)

#define LOGGED_BUFLEN_NOT_ALIGNED     0x01
#define LOGGED_GPU_OFFSET_NOT_ALIGNED 0x02

/*
 * Per-thread async submission state, stored in td->io_ops_data when
 * submit_mode is SUBMIT_BATCH or SUBMIT_STREAM.
 */
struct gpuaccel_async_data {
	const struct gpuaccel_async_ops *ops;
	void                     *ctx;        /* backend-private context */
	unsigned int              depth;
	unsigned int              queued;     /* staged since last commit */
	struct io_u             **queued_ios; /* staged io_us awaiting commit */
	struct gpuaccel_io_event *events;     /* reaped, indexed by ->event() */
	unsigned int              nevents;
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

		if (cur) {
			char *endptr;

			errno = 0;
			gpu_id = strtol(cur, &endptr, 10);
			if (errno != 0 || endptr == cur || *endptr != '\0') {
				log_err("invalid GPU ID: %s\n", cur);
				free(gpu_ids);
				return -1;
			}
		}

		free(gpu_ids);
	}

	return gpu_id;
}

int fio_gpuaccel_init(struct thread_data *td)
{
	struct gpuaccel_options *o = td->eo;
	const struct gpuaccel_backend *be = o->backend;
	int initialized;

	pthread_mutex_lock(be->running_lock);
	if (!*be->running) {
		assert(!*be->initialized);
		if (o->io_mode == IO_DIRECT) {
			/* only open the driver if this is the first worker thread */
			if (be->driver_open() != 0)
				log_err("%s driver_open failed\n", be->name);
			else
				*be->initialized = 1;
		}
	}
	(*be->running)++;
	initialized = *be->initialized;
	pthread_mutex_unlock(be->running_lock);

	if (o->io_mode == IO_DIRECT && !initialized)
		return 1;

	o->my_gpu_id = fio_gpuaccel_find_gpu_id(td);
	if (o->my_gpu_id < 0)
		return 1;

	dprint(FD_MEM, "Subjob %d uses GPU %d\n", td->subjob_number, o->my_gpu_id);
	if (be->set_device(o->my_gpu_id) != 0)
		return 1;

	/*
	 * POSIX I/O has no async path; force synchronous submission and warn if
	 * the user asked for batch or stream.
	 */
	if (o->io_mode == IO_POSIX &&
	    (o->submit_mode == SUBMIT_BATCH || o->submit_mode == SUBMIT_STREAM)) {
		log_err("%s: hipfile_mode batch/stream requires rocm_io=hipfile, "
			"forcing sync\n", be->name);
		o->submit_mode = SUBMIT_SYNC;
	}

	/* submit_mode 0 (backends without the option, e.g. libcufile) means sync */
	if (o->submit_mode == SUBMIT_BATCH || o->submit_mode == SUBMIT_STREAM) {
		const struct gpuaccel_async_ops *ops =
			o->submit_mode == SUBMIT_BATCH ? be->batch : be->stream;
		struct gpuaccel_async_data *ad;
		unsigned int depth = td->o.iodepth;

		if (!ops) {
			log_err("%s: submission mode %u not supported by backend\n",
				be->name, o->submit_mode);
			return 1;
		}

		ad = calloc(1, sizeof(*ad));
		if (!ad) {
			log_err("%s: async data calloc failed\n", be->name);
			return 1;
		}
		ad->ops = ops;
		ad->depth = depth;
		ad->events = calloc(depth, sizeof(*ad->events));
		ad->queued_ios = calloc(depth, sizeof(*ad->queued_ios));
		if (!ad->events || !ad->queued_ios) {
			log_err("%s: async data calloc failed\n", be->name);
			free(ad->queued_ios);
			free(ad->events);
			free(ad);
			return 1;
		}
		if (ops->setup(td, &ad->ctx, depth) != 0) {
			log_err("%s: async setup failed\n", be->name);
			free(ad->queued_ios);
			free(ad->events);
			free(ad);
			return 1;
		}
		td->io_ops_data = ad;
	}

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
			 * Data is being verified, copy the io_u buffer to GPU memory.
			 * This isn't done in the non-verify case because the data would
			 * already be in GPU memory in a normal direct io application.
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
		 * POSIX I/O is being used, the data has to be copied out of the
		 * GPU into a CPU buffer. GPU memory doesn't contain the actual
		 * data to write, copy the data to the junk buffer. The purpose
		 * of this is to add the overhead of memcpy() that would be
		 * present in a POSIX I/O GPU application.
		 */
		rc = be->memcpy(o->junk_buf + gpu_offset,
				((char*) o->gpu_mem_ptr) + gpu_offset,
				io_u->xfer_buflen, MEMCPY_DIRECTION_D2H);
		if (rc != 0) {
			log_err("DDIR_WRITE %s memcpy D2H failed\n", be->name);
			io_u->error = EIO;
		}
		if (be->sync_after_posix_write_copy) {
			rc = be->stream_sync();
			if (rc) {
				log_err("DDIR_WRITE stream synchronize failed\n");
				io_u->error = EIO;
			}
		}
	} else {
		log_err("Illegal %s IO type: %d\n", be->name, o->io_mode);
		assert(0);
		rc = -EINVAL;
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
		if (be->sync_after_verify_read_copy) {
			rc = be->stream_sync();
			if (rc) {
				log_err("DDIR_READ stream synchronize failed\n");
				io_u->error = EIO;
			}
		}
	} else {
		log_err("Illegal %s IO type: %d\n", be->name, o->io_mode);
		assert(0);
		rc = -EINVAL;
	}

	return rc;
}

enum fio_q_status fio_gpuaccel_queue(struct thread_data *td,
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
		 * There may be a better way to calculate gpu_offset. The intent is
		 * that gpu_offset equals the difference between io_u->xfer_buf and
		 * the page-aligned base address for io_u buffers.
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
					} else if (sz == 0) {
						io_u->error = EIO;
						log_err("%s Read: unexpected EOF\n", be->name);
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
					} else if (sz == 0) {
						io_u->error = EIO;
						log_err("pread: unexpected EOF\n");
					}
				} else {
					log_err("Illegal %s IO type: %d\n", be->name, o->io_mode);
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
					} else if (sz == 0) {
						io_u->error = EIO;
						log_err("%s Write: unexpected EOF\n", be->name);
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
					} else if (sz == 0) {
						io_u->error = EIO;
						log_err("pwrite: unexpected EOF\n");
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

enum fio_q_status fio_gpuaccel_queue_async(struct thread_data *td,
					   struct io_u *io_u)
{
	struct gpuaccel_options *o = td->eo;
	const struct gpuaccel_backend *be = o->backend;
	struct gpuaccel_async_data *ad = td->io_ops_data;
	void *file_handle;
	size_t gpu_offset;
	int rc;

	/* Sync submission (or POSIX) keeps the original blocking path. */
	if (o->submit_mode == SUBMIT_SYNC || !ad)
		return fio_gpuaccel_queue(td, io_u);

	fio_ro_check(td, io_u);

	switch (io_u->ddir) {
	case DDIR_SYNC:
		rc = fsync(io_u->file->fd);
		if (rc != 0) {
			io_u->error = errno;
			log_err("fsync: err=%d\n", errno);
			td_verror(td, io_u->error, "fsync");
		}
		return FIO_Q_COMPLETED;
	case DDIR_DATASYNC:
		rc = fdatasync(io_u->file->fd);
		if (rc != 0) {
			io_u->error = errno;
			log_err("fdatasync: err=%d\n", errno);
			td_verror(td, io_u->error, "fdatasync");
		}
		return FIO_Q_COMPLETED;
	case DDIR_READ:
	case DDIR_WRITE:
		break;
	default:
		io_u->error = EINVAL;
		td_verror(td, io_u->error, "xfer");
		return FIO_Q_COMPLETED;
	}

	file_handle = FILE_ENG_DATA(io_u->file);
	if (file_handle == NULL) {
		io_u->error = EINVAL;
		td_verror(td, EINVAL, "xfer");
		return FIO_Q_COMPLETED;
	}

	if (ad->queued >= ad->depth)
		return FIO_Q_BUSY;

	gpu_offset = io_u->index * io_u->xfer_buflen;
	assert(gpu_offset + io_u->xfer_buflen <= o->total_mem);

	if (io_u->ddir == DDIR_WRITE) {
		fio_gpuaccel_pre_write(td, o, io_u, gpu_offset);
		if (io_u->error != 0) {
			td_verror(td, io_u->error, "xfer");
			return FIO_Q_COMPLETED;
		}
	}

	rc = ad->ops->prep(ad->ctx, io_u, file_handle, o->gpu_mem_ptr,
			   io_u->xfer_buflen, io_u->offset, gpu_offset,
			   io_u->ddir);
	if (rc != 0) {
		io_u->error = EIO;
		log_err("%s: async prep failed\n", be->name);
		td_verror(td, io_u->error, "xfer");
		return FIO_Q_COMPLETED;
	}

	ad->queued_ios[ad->queued++] = io_u;
	return FIO_Q_QUEUED;
}

int fio_gpuaccel_commit(struct thread_data *td)
{
	struct gpuaccel_options *o = td->eo;
	const struct gpuaccel_backend *be = o->backend;
	struct gpuaccel_async_data *ad = td->io_ops_data;
	int rc;

	if (!ad || !ad->queued)
		return 0;

	rc = ad->ops->submit(ad->ctx, ad->queued);
	if (rc != 0) {
		log_err("%s: async submit failed\n", be->name);
		td_verror(td, EIO, "commit");
		return -1;
	}

	/*
	 * issue_time is stamped generically by td_io_queue() since we don't set
	 * FIO_ASYNCIO_SETS_ISSUE_TIME (the engine serves both sync-completed and
	 * async-queued io_us through one ->queue).
	 */
	io_u_mark_submit(td, ad->queued);
	ad->queued = 0;
	return 0;
}

int fio_gpuaccel_getevents(struct thread_data *td, unsigned int min,
			   unsigned int max, const struct timespec *t)
{
	struct gpuaccel_options *o = td->eo;
	struct gpuaccel_async_data *ad = td->io_ops_data;
	int n, i;

	if (!ad)
		return 0;

	if (max > ad->depth)
		max = ad->depth;

	n = ad->ops->reap(ad->ctx, min, max, ad->events, t);
	if (n < 0) {
		td_verror(td, EIO, "getevents");
		return -1;
	}

	for (i = 0; i < n; i++) {
		struct gpuaccel_io_event *ev = &ad->events[i];
		struct io_u *io_u = ev->io_u;
		size_t gpu_offset;

		if (ev->error != 0) {
			io_u->error = ev->error;
			continue;
		}

		if (ev->ret < 0) {
			io_u->error = EIO;
			log_err("%s async %s: err=%zd:%s\n", o->backend->name,
				io_u->ddir == DDIR_READ ? "read" : "write",
				ev->ret, o->backend->op_error_string(-ev->ret));
			continue;
		}

		if ((size_t)ev->ret < io_u->xfer_buflen) {
			io_u->resid = io_u->xfer_buflen - ev->ret;
			io_u->error = 0;
			continue;
		}

		if (io_u->ddir == DDIR_READ) {
			gpu_offset = io_u->index * io_u->xfer_buflen;
			fio_gpuaccel_post_read(td, o, io_u, gpu_offset);
		}
	}

	ad->nevents = n;
	return n;
}

struct io_u *fio_gpuaccel_event(struct thread_data *td, int event)
{
	struct gpuaccel_async_data *ad = td->io_ops_data;

	return ad->events[event].io_u;
}

int fio_gpuaccel_open_file(struct thread_data *td, struct fio_file *f)
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

int fio_gpuaccel_close_file(struct thread_data *td, struct fio_file *f)
{
	void *handle = FILE_ENG_DATA(f);
	int rc;
	struct gpuaccel_options *o = td->eo;
	const struct gpuaccel_backend *be = o->backend;

	if (handle != NULL) {
		be->file_handle_deregister(handle);
		FILE_SET_ENG_DATA(f, NULL);
	}

	rc = generic_close_file(td, f);

	return rc;
}

int fio_gpuaccel_iomem_alloc(struct thread_data *td, size_t total_mem)
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
	if (be->sync_after_memset) {
		rc = be->stream_sync();
		if (rc != 0)
			goto exit_error;
	}
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

void fio_gpuaccel_iomem_free(struct thread_data *td)
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

void fio_gpuaccel_cleanup(struct thread_data *td)
{
	struct gpuaccel_options *o = td->eo;
	const struct gpuaccel_backend *be = o->backend;
	struct gpuaccel_async_data *ad = td->io_ops_data;

	if (ad) {
		ad->ops->destroy(td, ad->ctx);
		free(ad->queued_ios);
		free(ad->events);
		free(ad);
		td->io_ops_data = NULL;
	}

	pthread_mutex_lock(be->running_lock);
	(*be->running)--;
	assert(*be->running >= 0);
	if (!*be->running) {
		/*
		 * Only close the driver if initialized and
		 * this is the last worker thread.
		 */
		if (o->io_mode == IO_DIRECT && *be->initialized)
			be->driver_close();
		*be->initialized = 0;
	}
	pthread_mutex_unlock(be->running_lock);
}
