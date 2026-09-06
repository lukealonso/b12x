/* Native descriptor batches. Python retains file and tensor owners until return. */
#include <limits.h>
#include <time.h>

#define BATCH_CHUNK_BYTES (64 << 20)
typedef struct {
    int fd;
    int64_t offset, bytes;
    char *destination;
    bool expand_bf16;
    bool host_copy;
    int64_t rows, source_stride, destination_stride;
} read_job_t;

typedef struct batch_executor batch_executor_t;
typedef struct {
    batch_executor_t *executor;
    PyObject *owner;
    direct_reader_t *reader;
    pthread_t thread;
} batch_worker_t;

struct batch_executor {
    pthread_mutex_t mutex;
    pthread_cond_t ready, done;
    batch_worker_t workers[16];
    int count, running;
    bool stopping, active;
    uint64_t generation, batches, descriptors;
    double execution_seconds;
    read_job_t *jobs;
    size_t job_count, next_job;
    failure_t failure;
};

static void *batch_worker(void *opaque) {
    batch_worker_t *worker = opaque;
    batch_executor_t *executor = worker->executor;
    uint64_t generation = 0;
    pthread_mutex_lock(&executor->mutex);
    for (;;) {
        while (!executor->stopping && executor->generation == generation)
            pthread_cond_wait(&executor->ready, &executor->mutex);
        if (executor->stopping) break;
        generation = executor->generation;
        while (executor->next_job < executor->job_count && !executor->failure.message[0]) {
            read_job_t job = executor->jobs[executor->next_job++];
            pthread_mutex_unlock(&executor->mutex);
            failure_t failure = {{0}};
            bool success = true;
            if (job.host_copy) memcpy(job.destination, (void *)(uintptr_t)job.offset, job.bytes);
            else success = direct_read_rows(worker->reader, job.fd, job.offset,
                job.bytes, job.rows, job.source_stride, job.destination_stride,
                job.destination, job.expand_bf16, &failure);
            if (success && job.host_copy && job.expand_bf16)
                expand_bf16_inplace(job.destination, job.bytes);
            pthread_mutex_lock(&executor->mutex);
            if (!success && !executor->failure.message[0]) executor->failure = failure;
        }
        if (--executor->running == 0) pthread_cond_signal(&executor->done);
    }
    pthread_mutex_unlock(&executor->mutex);
    return NULL;
}

static void release_batch(batch_executor_t *executor) {
    pthread_mutex_lock(&executor->mutex);
    executor->stopping = true;
    pthread_cond_broadcast(&executor->ready);
    pthread_mutex_unlock(&executor->mutex);
    for (int i = 0; i < executor->count; i++)
        pthread_join(executor->workers[i].thread, NULL);
    for (int i = 0; i < 16; i++) Py_XDECREF(executor->workers[i].owner);
    pthread_cond_destroy(&executor->ready);
    pthread_cond_destroy(&executor->done);
    pthread_mutex_destroy(&executor->mutex);
    free(executor);
}

static void delete_batch(PyObject *capsule) {
    batch_executor_t *executor = PyCapsule_GetPointer(capsule, "b12x.batch_executor");
    if (executor) release_batch(executor);
}

static PyObject *py_batch_executor(PyObject *self, PyObject *args) {
    (void)self;
    int device, workers;
    if (!PyArg_ParseTuple(args, "ii", &device, &workers)) return NULL;
    if (workers < 1 || workers > 16)
        return PyErr_Format(PyExc_ValueError, "io_threads must be between 1 and 16");
    batch_executor_t *executor = calloc(1, sizeof(*executor));
    if (!executor) return PyErr_NoMemory();
    pthread_mutex_init(&executor->mutex, NULL);
    pthread_cond_init(&executor->ready, NULL);
    pthread_cond_init(&executor->done, NULL);
    for (int i = 0; i < workers; i++) {
        batch_worker_t *worker = &executor->workers[i];
        worker->executor = executor;
        PyObject *reader_args = Py_BuildValue("(i)", device);
        worker->owner = reader_args ? py_direct_reader(NULL, reader_args) : NULL;
        Py_XDECREF(reader_args);
        if (!worker->owner) {
            release_batch(executor);
            return NULL;
        }
        worker->reader = PyCapsule_GetPointer(worker->owner, "b12x.direct_reader");
        int status = pthread_create(&worker->thread, NULL, batch_worker, worker);
        if (status) {
            release_batch(executor);
            return PyErr_Format(PyExc_RuntimeError, "pthread_create: %s", strerror(status));
        }
        executor->count++;
    }
    PyObject *capsule = PyCapsule_New(executor, "b12x.batch_executor", delete_batch);
    if (!capsule) release_batch(executor);
    return capsule;
}

static int job_destination_order(const void *a, const void *b) {
    uintptr_t x = (uintptr_t)((const read_job_t *)a)->destination;
    uintptr_t y = (uintptr_t)((const read_job_t *)b)->destination;
    return (x > y) - (x < y);
}

static int job_source_order(const void *a, const void *b) {
    const read_job_t *x = a, *y = b;
    if (x->fd != y->fd) return (x->fd > y->fd) - (x->fd < y->fd);
    return (x->offset > y->offset) - (x->offset < y->offset);
}

static PyObject *py_batch_execute(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule;
    Py_buffer records;
    unsigned long long stream;
    if (!PyArg_ParseTuple(args, "Oy*K", &capsule, &records, &stream)) return NULL;
    batch_executor_t *executor = PyCapsule_GetPointer(capsule, "b12x.batch_executor");
    if (!executor) { PyBuffer_Release(&records); return NULL; }
    failure_t failure = {{0}};
    read_job_t *jobs = NULL;
    size_t count = 0, capacity = 0;
    if (records.len % (8 * sizeof(uint64_t))) {
        snprintf(failure.message, sizeof(failure.message), "invalid read descriptor byte size");
        goto done;
    }
    size_t descriptors = records.len / (8 * sizeof(uint64_t));
    for (size_t i = 0; i < descriptors; i++) {
        uint64_t record[8];
        memcpy(record, (char *)records.buf + i * sizeof(record), sizeof(record));
        uint64_t fd = record[0], offset = record[1], bytes = record[2];
        uint64_t pointer = record[3], transform = record[4];
        uint64_t rows = record[5], source_stride = record[6], destination_stride = record[7];
        bool expand_bf16 = transform == 1;
        if (fd > INT_MAX || transform > 2 || offset > INT64_MAX ||
            bytes > (uint64_t)INT64_MAX - offset ||
            (expand_bf16 && (bytes % 2 || bytes > INT64_MAX / 2)) ||
            (transform == 2 && ((!offset && bytes) || rows != 1)) || !rows || rows > INT64_MAX ||
            source_stride > INT64_MAX || destination_stride > INT64_MAX ||
            (source_stride && rows - 1 > ((uint64_t)INT64_MAX - offset - bytes) / source_stride) ||
            (destination_stride && rows - 1 > ((uint64_t)INT64_MAX - bytes * (1 + expand_bf16)) / destination_stride)) {
            snprintf(failure.message, sizeof(failure.message), "invalid read descriptor");
            goto done;
        }
        if (rows > 1 && destination_stride < bytes * (1 + expand_bf16)) {
            snprintf(failure.message, sizeof(failure.message), "overlapping batch destinations need an explicit dependency");
            goto done;
        }
        pthread_mutex_lock(&pool_mutex);
        uint64_t destination_extent = (rows - 1) * destination_stride + bytes * (1 + expand_bf16);
        storage_t *storage = find_segment(pointer, destination_extent);
        char *destination = storage && storage->locked ?
            (char *)storage->base + (pointer - (uintptr_t)storage->gpu) : NULL;
        pthread_mutex_unlock(&pool_mutex);
        if (!destination) {
            snprintf(failure.message, sizeof(failure.message), "batch destination is not owned locked storage");
            goto done;
        }
        while (rows && bytes) {
            if (count == capacity) {
                size_t next = capacity ? capacity * 2 : 1024;
                read_job_t *grown = realloc(jobs, next * sizeof(*jobs));
                if (!grown) { snprintf(failure.message, sizeof(failure.message), "read descriptor allocation failed"); goto done; }
                jobs = grown;
                capacity = next;
            }
            int64_t chunk = bytes > BATCH_CHUNK_BYTES ? BATCH_CHUNK_BYTES : (int64_t)bytes;
            uint64_t chunk_rows = 1;
            if (bytes <= BATCH_CHUNK_BYTES && destination_stride == bytes * (1 + expand_bf16)) {
                uint64_t stride = source_stride > destination_stride ? source_stride : destination_stride;
                chunk_rows = stride ? BATCH_CHUNK_BYTES / stride : 1;
                if (!chunk_rows) chunk_rows = 1;
                if (chunk_rows > rows) chunk_rows = rows;
            }
            jobs[count++] = (read_job_t){(int)fd, (int64_t)offset, chunk, destination,
                expand_bf16, transform == 2, (int64_t)chunk_rows,
                (int64_t)source_stride, (int64_t)destination_stride};
            if (chunk < (int64_t)bytes) {
                /* Large contiguous rows can still be divided into independent jobs. */
                if (rows != 1) {
                    snprintf(failure.message, sizeof(failure.message), "strided rows exceed batch chunk size");
                    goto done;
                }
                offset += chunk;
                bytes -= chunk;
                destination += chunk * (1 + expand_bf16);
            } else {
                rows -= chunk_rows;
                if (rows) {
                    offset += chunk_rows * source_stride;
                    destination += chunk_rows * destination_stride;
                }
            }
        }
    }
    qsort(jobs, count, sizeof(*jobs), job_destination_order);
    for (size_t i = 1; i < count; i++) {
        read_job_t *previous = &jobs[i - 1];
        if ((uintptr_t)previous->destination + previous->bytes * (1 + previous->expand_bf16) +
            (previous->rows - 1) * previous->destination_stride >
            (uintptr_t)jobs[i].destination) {
            snprintf(failure.message, sizeof(failure.message), "overlapping batch destinations need an explicit dependency");
            goto done;
        }
    }
    qsort(jobs, count, sizeof(*jobs), job_source_order);
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&executor->mutex);
    if (executor->active) {
        snprintf(failure.message, sizeof(failure.message), "batch executor is already active");
    } else if (cuda_ok(cudaStreamSynchronize((cudaStream_t)(uintptr_t)stream),
                       "synchronize batch destinations", &failure)) {
        executor->active = true;
        executor->jobs = jobs;
        executor->job_count = count;
        executor->next_job = 0;
        executor->running = executor->count;
        executor->failure.message[0] = '\0';
        struct timespec start, stop;
        clock_gettime(CLOCK_MONOTONIC, &start);
        executor->generation++;
        pthread_cond_broadcast(&executor->ready);
        while (executor->running) pthread_cond_wait(&executor->done, &executor->mutex);
        clock_gettime(CLOCK_MONOTONIC, &stop);
        executor->execution_seconds += stop.tv_sec - start.tv_sec +
                                       (stop.tv_nsec - start.tv_nsec) * 1e-9;
        failure = executor->failure;
        executor->jobs = NULL;
        executor->active = false;
        executor->batches++;
        executor->descriptors += descriptors;
    }
    pthread_mutex_unlock(&executor->mutex);
    Py_END_ALLOW_THREADS
done:
    free(jobs);
    PyBuffer_Release(&records);
    if (failure.message[0]) return PyErr_Format(PyExc_RuntimeError, "%s", failure.message);
    Py_RETURN_NONE;
}

static PyObject *py_batch_stats(PyObject *self, PyObject *capsule) {
    (void)self;
    batch_executor_t *executor = PyCapsule_GetPointer(capsule, "b12x.batch_executor");
    if (!executor) return NULL;
    pthread_mutex_lock(&executor->mutex);
    if (executor->active) {
        pthread_mutex_unlock(&executor->mutex);
        return PyErr_Format(PyExc_RuntimeError, "batch stats require completed reads");
    }
    direct_reader_t total = {0};
    for (int i = 0; i < executor->count; i++) {
        direct_reader_t *reader = executor->workers[i].reader;
        total.physical_bytes += reader->physical_bytes;
        total.destination_bytes += reader->destination_bytes;
        total.realigned_bytes += reader->realigned_bytes;
        total.inplace_aligned_bytes += reader->inplace_aligned_bytes;
        total.strided_copy_bytes += reader->strided_copy_bytes;
        total.reads += reader->reads;
    }
    PyObject *result = Py_BuildValue("{s:K,s:K,s:K,s:K,s:K,s:K,s:i,s:K,s:K,s:d}",
        "physical_bytes", (unsigned long long)total.physical_bytes,
        "destination_bytes", (unsigned long long)total.destination_bytes,
        "realigned_bytes", (unsigned long long)total.realigned_bytes,
        "inplace_aligned_bytes", (unsigned long long)total.inplace_aligned_bytes,
        "strided_copy_bytes", (unsigned long long)total.strided_copy_bytes,
        "reads", (unsigned long long)total.reads,
        "scratch_bytes", executor->count * IO_SCRATCH_BYTES,
        "batches", (unsigned long long)executor->batches,
        "descriptors", (unsigned long long)executor->descriptors,
        "execution_seconds", executor->execution_seconds);
    pthread_mutex_unlock(&executor->mutex);
    return result;
}
