/* Included by _storage.c. Every payload read requires an O_DIRECT descriptor. */
#include <fcntl.h>

#define IO_ALIGNMENT 4096
#define IO_SCRATCH_BYTES (8 << 20)

typedef struct {
    storage_t *scratch;
    pthread_mutex_t mutex;
    uint64_t physical_bytes;
    uint64_t destination_bytes;
    uint64_t realigned_bytes;
    uint64_t inplace_aligned_bytes;
    uint64_t strided_copy_bytes;
    uint64_t reads;
} direct_reader_t;

static void delete_reader(PyObject *capsule) {
    direct_reader_t *reader = PyCapsule_GetPointer(capsule, "b12x.direct_reader");
    if (!reader) return;
    release_storage(reader->scratch);
    pthread_mutex_destroy(&reader->mutex);
    free(reader);
}

static PyObject *py_direct_reader(PyObject *self, PyObject *args) {
    (void)self;
    int device;
    if (!PyArg_ParseTuple(args, "i", &device)) return NULL;
    direct_reader_t *reader = calloc(1, sizeof(*reader));
    if (!reader) return PyErr_NoMemory();
    failure_t failure = {{0}};
    reader->scratch = read_storage(-1, 0, IO_SCRATCH_BYTES, "registered", device, &failure);
    if (reader->scratch && mlock(reader->scratch->base, IO_SCRATCH_BYTES) != 0) {
        system_error(&failure, "mlock direct I/O scratch");
        release_storage(reader->scratch);
        reader->scratch = NULL;
    }
    if (!reader->scratch) {
        free(reader);
        return PyErr_Format(PyExc_RuntimeError, "%s", failure.message);
    }
    reader->scratch->locked = true;
    pthread_mutex_init(&reader->mutex, NULL);
    PyObject *capsule = PyCapsule_New(reader, "b12x.direct_reader", delete_reader);
    if (!capsule) {
        release_storage(reader->scratch);
        pthread_mutex_destroy(&reader->mutex);
        free(reader);
    }
    return capsule;
}

static bool validate_direct_range(int fd, int64_t offset, int64_t bytes,
                                  failure_t *failure) {
    struct stat status;
    int flags = fcntl(fd, F_GETFL);
    if (flags < 0 || !(flags & O_DIRECT)) {
        snprintf(failure->message, sizeof(failure->message), "direct reader requires O_DIRECT");
        return false;
    }
    if (fstat(fd, &status) != 0) {
        system_error(failure, "fstat direct input");
        return false;
    }
    if (offset < 0 || bytes < 0 || offset > INT64_MAX - bytes ||
        !S_ISREG(status.st_mode) || offset + bytes > status.st_size) {
        snprintf(failure->message, sizeof(failure->message), "invalid direct input file range");
        return false;
    }
    return true;
}

static bool direct_read_range(direct_reader_t *reader, int fd, int64_t offset,
                              int64_t bytes, char *destination, bool allow_direct,
                              failure_t *failure) {
    if (!validate_direct_range(fd, offset, bytes, failure)) return false;
    while (bytes) {
        int64_t aligned_offset = offset & ~(int64_t)(IO_ALIGNMENT - 1);
        size_t delta = offset - aligned_offset;
        size_t address_delta = (uintptr_t)destination % IO_ALIGNMENT;
        bool direct = allow_direct && address_delta == 0 &&
                      bytes >= (delta ? 2 * IO_ALIGNMENT : IO_ALIGNMENT);
        size_t payload = bytes < IO_SCRATCH_BYTES - (int64_t)delta ?
                         (size_t)bytes : IO_SCRATCH_BYTES - delta;
        size_t length;
        if (direct) {
            length = ((size_t)(bytes < IO_SCRATCH_BYTES ? bytes : IO_SCRATCH_BYTES)) &
                     ~(size_t)(IO_ALIGNMENT - 1);
            payload = length - (delta ? IO_ALIGNMENT : 0);
        } else {
            if (allow_direct && address_delta && payload > IO_ALIGNMENT - address_delta)
                payload = IO_ALIGNMENT - address_delta;
            length = (payload + delta + IO_ALIGNMENT - 1) & ~(size_t)(IO_ALIGNMENT - 1);
        }
        void *buffer = direct ? (void *)destination : reader->scratch->base;
        ssize_t count;
        do {
            count = pread(fd, buffer, length, aligned_offset);
        } while (count < 0 && errno == EINTR);
        if (count < 0) {
            system_error(failure, "O_DIRECT pread (no buffered fallback)");
            return false;
        }
        if ((size_t)count < delta + payload) {
            snprintf(failure->message, sizeof(failure->message), "short O_DIRECT read");
            return false;
        }
        reader->physical_bytes += count;
        reader->reads++;
        if (direct) {
            reader->destination_bytes += payload;
            if (delta) {
                /* The aligned window fits within the remaining destination.
                   Shift its payload in place; the next read replaces lookahead. */
                memmove(destination, destination + delta, payload);
                reader->inplace_aligned_bytes += payload;
            }
        }
        else {
            memcpy(destination, (char *)buffer + delta, payload);
            reader->realigned_bytes += payload;
        }
        destination += payload;
        offset += payload;
        bytes -= payload;
    }
    return true;
}

static PyObject *py_direct_bytes(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule;
    int fd;
    long long offset, bytes;
    if (!PyArg_ParseTuple(args, "OiLL", &capsule, &fd, &offset, &bytes)) return NULL;
    direct_reader_t *reader = PyCapsule_GetPointer(capsule, "b12x.direct_reader");
    if (!reader) return NULL;
    if (bytes < 0 || bytes > 100 * (1 << 20))
        return PyErr_Format(PyExc_ValueError, "header/metadata read exceeds 100 MiB");
    PyObject *result = PyBytes_FromStringAndSize(NULL, bytes);
    if (!result) return NULL;
    bool success;
    failure_t failure = {{0}};
    char *destination = PyBytes_AS_STRING(result);
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&reader->mutex);
    success = direct_read_range(reader, fd, offset, bytes, destination, false, &failure);
    pthread_mutex_unlock(&reader->mutex);
    Py_END_ALLOW_THREADS
    if (!success) {
        Py_DECREF(result);
        return PyErr_Format(PyExc_RuntimeError, "%s", failure.message);
    }
    return result;
}

static void expand_bf16_inplace(char *destination, int64_t bytes) {
    for (int64_t i = bytes / 2; i > 0;) {
        --i;
        uint16_t bf16;
        memcpy(&bf16, destination + 2 * i, sizeof(bf16));
        uint32_t fp32 = (uint32_t)bf16 << 16;
        memcpy(destination + 4 * i, &fp32, sizeof(fp32));
    }
}

static bool direct_read_rows(direct_reader_t *reader, int fd, int64_t offset,
                             int64_t bytes, int64_t rows, int64_t source_stride,
                             int64_t destination_stride, char *destination,
                             bool expand_bf16, failure_t *failure) {
    int64_t extent = (rows - 1) * source_stride + bytes;
    if (!validate_direct_range(fd, offset, extent, failure)) return false;
    while (rows) {
        int64_t aligned = offset & ~(int64_t)(IO_ALIGNMENT - 1);
        size_t delta = offset - aligned;
        if (rows == 1 || bytes >= IO_ALIGNMENT || !source_stride ||
            source_stride > IO_SCRATCH_BYTES - (int64_t)delta - bytes) {
            if (!direct_read_range(reader, fd, offset, bytes, destination, true, failure))
                return false;
            if (expand_bf16) expand_bf16_inplace(destination, bytes);
            rows--;
            if (rows) {
                offset += source_stride;
                destination += destination_stride;
            }
            continue;
        }
        int64_t count_rows = 1 + (IO_SCRATCH_BYTES - (int64_t)delta - bytes) / source_stride;
        if (count_rows > rows) count_rows = rows;
        size_t payload = (count_rows - 1) * source_stride + bytes;
        size_t length = (delta + payload + IO_ALIGNMENT - 1) & ~(size_t)(IO_ALIGNMENT - 1);
        ssize_t count;
        do {
            count = pread(fd, reader->scratch->base, length, aligned);
        } while (count < 0 && errno == EINTR);
        if (count < 0) {
            system_error(failure, "O_DIRECT strided pread (no buffered fallback)");
            return false;
        }
        if ((size_t)count < delta + payload) {
            snprintf(failure->message, sizeof(failure->message), "short O_DIRECT strided read");
            return false;
        }
        reader->physical_bytes += count;
        reader->reads++;
        for (int64_t row = 0; row < count_rows; row++) {
            char *target = destination + row * destination_stride;
            memcpy(target, (char *)reader->scratch->base + delta + row * source_stride, bytes);
            if (expand_bf16) expand_bf16_inplace(target, bytes);
        }
        reader->strided_copy_bytes += count_rows * bytes;
        rows -= count_rows;
        if (rows) {
            offset += count_rows * source_stride;
            destination += count_rows * destination_stride;
        }
    }
    return true;
}

static PyObject *direct_into(PyObject *args, bool expand_bf16) {
    PyObject *capsule;
    int fd;
    long long offset, bytes;
    unsigned long long pointer, stream;
    if (!PyArg_ParseTuple(args, "OiLLKK", &capsule, &fd, &offset, &bytes, &pointer, &stream)) return NULL;
    direct_reader_t *reader = PyCapsule_GetPointer(capsule, "b12x.direct_reader");
    if (!reader) return NULL;
    if (bytes < 0 || (expand_bf16 && (bytes % 2 || bytes > INT64_MAX / 2)))
        return PyErr_Format(PyExc_ValueError, "invalid direct input byte count");
    int64_t destination_bytes = expand_bf16 ? bytes * 2 : bytes;
    pthread_mutex_lock(&pool_mutex);
    storage_t *storage = find_segment(pointer, destination_bytes);
    char *destination = storage && storage->locked ?
        (char *)storage->base + (pointer - (uintptr_t)storage->gpu) : NULL;
    pthread_mutex_unlock(&pool_mutex);
    if (!destination) return PyErr_Format(PyExc_ValueError, "destination is not owned shared pool storage");
    failure_t failure = {{0}};
    bool success;
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&reader->mutex);
    success = cuda_ok(cudaStreamSynchronize((cudaStream_t)(uintptr_t)stream),
                      "synchronize direct I/O destination", &failure) &&
              direct_read_range(reader, fd, offset, bytes, destination, true, &failure);
    if (success && expand_bf16) {
        /* Widen backwards before publishing the allocation to GPU consumers.
           memcpy avoids aliasing and alignment assumptions for tensor views. */
        expand_bf16_inplace(destination, bytes);
    }
    pthread_mutex_unlock(&reader->mutex);
    Py_END_ALLOW_THREADS
    if (!success) return PyErr_Format(PyExc_RuntimeError, "%s", failure.message);
    Py_RETURN_NONE;
}

static PyObject *py_direct_into(PyObject *self, PyObject *args) {
    (void)self;
    return direct_into(args, false);
}

static PyObject *py_direct_bf16_into_fp32(PyObject *self, PyObject *args) {
    (void)self;
    return direct_into(args, true);
}

static PyObject *py_direct_stats(PyObject *self, PyObject *capsule) {
    (void)self;
    direct_reader_t *reader = PyCapsule_GetPointer(capsule, "b12x.direct_reader");
    if (!reader) return NULL;
    pthread_mutex_lock(&reader->mutex);
    PyObject *result = Py_BuildValue("{s:K,s:K,s:K,s:K,s:K,s:i}",
        "physical_bytes", (unsigned long long)reader->physical_bytes,
        "destination_bytes", (unsigned long long)reader->destination_bytes,
        "realigned_bytes", (unsigned long long)reader->realigned_bytes,
        "inplace_aligned_bytes", (unsigned long long)reader->inplace_aligned_bytes,
        "reads", (unsigned long long)reader->reads, "scratch_bytes", IO_SCRATCH_BYTES);
    pthread_mutex_unlock(&reader->mutex);
    return result;
}
