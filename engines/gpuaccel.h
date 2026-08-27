#ifndef FIO_GPUACCEL_H
#define FIO_GPUACCEL_H

#include <pthread.h>
#include <unistd.h>

#define GPU_ID_SEP ":"

enum fio_q_status;
struct thread_data;
struct io_u;
struct fio_file;

enum {
	IO_DIRECT    = 1,
	IO_POSIX     = 2
};

enum {
	MEMCPY_DIRECTION_H2D = 1,
	MEMCPY_DIRECTION_D2H = 2
};

enum {
	SUBMIT_SYNC   = 1,	/* synchronous ->queue (default) */
	SUBMIT_BATCH  = 2,	/* async via hipfile batch API */
	SUBMIT_STREAM = 3	/* async via hipfile stream API */
};

/* One reaped async completion, handed back to the gpuaccel glue by a backend. */
struct gpuaccel_io_event {
	struct io_u *io_u;
	ssize_t      ret;	/* bytes transferred, or negative op-error code */
	int          error;	/* errno to set on io_u, 0 if none */
};

/*
 * Async submission vtable, implemented once per submission model (batch,
 * stream) by a gpuaccel backend. The gpuaccel layer owns the fio-facing
 * queue/commit/getevents/event glue and the verify/posix memcpy dance; a
 * backend only stages, submits, and reaps I/O.
 */
struct gpuaccel_async_ops {
	int  (*setup)(struct thread_data *td, void **ctx, unsigned int depth);
	void (*destroy)(struct thread_data *td, void *ctx);
	/* stage one io_u for submission (called from ->queue) */
	int  (*prep)(void *ctx, struct io_u *io_u, void *handle, void *mem,
		     size_t size, unsigned long long file_off, size_t mem_off,
		     int ddir);
	/* submit all staged ops (called from ->commit) */
	int  (*submit)(void *ctx, unsigned int nr);
	/* reap min..max completions into events[]; return count or -1 on error */
	int  (*reap)(void *ctx, unsigned int min, unsigned int max,
		     struct gpuaccel_io_event *events, const struct timespec *t);
};

struct gpuaccel_backend {
	const char *name;

	int sync_after_posix_write_copy;
	int sync_after_verify_read_copy;
	int sync_after_memset;

	int *running;
	int *initialized;
	pthread_mutex_t *running_lock;

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

	const struct gpuaccel_async_ops *batch;  /* NULL if unsupported */
	const struct gpuaccel_async_ops *stream; /* NULL if unsupported */
};

struct gpuaccel_options {
	struct thread_data *td;
	char               *gpu_ids;            /* colon-separated list of GPU ids,
					                           one per job */
	void               *gpu_mem_ptr;        /* GPU memory */
	void               *junk_buf;           /* buffer to simulate cudaMemcpy
					                           with posix I/O write */
	int                 my_gpu_id;          /* GPU id to use for this job */
	unsigned int        io_mode;            /* Type of I/O to use */
	unsigned int        submit_mode;        /* sync/batch/stream submission */
	size_t              total_mem;          /* size for gpu_mem_ptr and junk_buf */
	int                 logged;             /* bitmask of log messages that have
					                           been output, prevent flood */
	const struct gpuaccel_backend *backend; /* GPU accelerator backend vtable */
};

int fio_gpuaccel_init(struct thread_data *td);
void fio_gpuaccel_cleanup(struct thread_data *td);
enum fio_q_status fio_gpuaccel_queue(struct thread_data *td, struct io_u *io_u);
int fio_gpuaccel_open_file(struct thread_data *td, struct fio_file *f);
int fio_gpuaccel_close_file(struct thread_data *td, struct fio_file *f);
int fio_gpuaccel_iomem_alloc(struct thread_data *td, size_t total_mem);
void fio_gpuaccel_iomem_free(struct thread_data *td);

/* async submission path (batch/stream); sync mode still uses fio_gpuaccel_queue */
enum fio_q_status fio_gpuaccel_queue_async(struct thread_data *td,
					   struct io_u *io_u);
int fio_gpuaccel_commit(struct thread_data *td);
int fio_gpuaccel_getevents(struct thread_data *td, unsigned int min,
			   unsigned int max, const struct timespec *t);
struct io_u *fio_gpuaccel_event(struct thread_data *td, int event);

#endif
