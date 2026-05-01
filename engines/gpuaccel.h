#ifndef FIO_GPUACCEL_H
#define FIO_GPUACCEL_H

#include <pthread.h>
#include "../fio.h"

#define GPUACCEL_GPU_ID_SEP ":"

#define GPUACCEL_LOGGED_BUFLEN_NOT_ALIGNED     0x01
#define GPUACCEL_LOGGED_GPU_OFFSET_NOT_ALIGNED 0x02

#define GPUACCEL_IO_DIRECT_MODE 1
#define GPUACCEL_IO_POSIX_MODE  2

#define GPUACCEL_MEMCPY_H2D 1
#define GPUACCEL_MEMCPY_D2H 2

struct gpuaccel_backend {
	const char *name;
	const char *direct_io_name;
	int *running;
	int *initialized;
	pthread_mutex_t *running_lock;

	int (*driver_open)(void);
	void (*driver_close)(void);

	int (*set_device)(int gpu_id);
	int (*mem_alloc)(void **mem, size_t size);
	int (*mem_free)(void *mem);
	int (*mem_set)(void *mem, int value, size_t size);
	int (*mem_copy)(void *dst, const void *src, size_t size, int direction);
	int (*stream_sync)(void);

	int (*file_handle_register)(int fd, void **handle);
	void (*file_handle_deregister)(void *handle);

	int (*buf_register)(void *mem, size_t size);
	void (*buf_deregister)(void *mem);

	ssize_t (*direct_read)(void *handle, void *mem, size_t size,
			      unsigned long long file_offset, size_t mem_offset);
	ssize_t (*direct_write)(void *handle, const void *mem, size_t size,
			       unsigned long long file_offset, size_t mem_offset);

	const char *(*op_error_string)(int error_code);

	int sync_after_posix_write_copy;
	int sync_after_verify_read_copy;
	int sync_after_memset;
};

struct gpuaccel_options {
	struct thread_data *td;
	char *gpu_ids;
	void *gpu_mem_ptr;
	void *junk_buf;
	int my_gpu_id;
	unsigned int io_mode;
	size_t total_mem;
	int logged;
	const struct gpuaccel_backend *backend;
};

int fio_gpuaccel_init(struct thread_data *td);
void fio_gpuaccel_cleanup(struct thread_data *td);
enum fio_q_status fio_gpuaccel_queue(struct thread_data *td, struct io_u *io_u);
int fio_gpuaccel_open_file(struct thread_data *td, struct fio_file *f);
int fio_gpuaccel_close_file(struct thread_data *td, struct fio_file *f);
int fio_gpuaccel_iomem_alloc(struct thread_data *td, size_t total_mem);
void fio_gpuaccel_iomem_free(struct thread_data *td);

#endif
