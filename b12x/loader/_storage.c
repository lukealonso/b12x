#define _GNU_SOURCE
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cuda_runtime_api.h>
#include <dlpack.h>

#include <errno.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef char require_64_bit_offsets[(sizeof(off_t) == 8 && sizeof(size_t) == 8) ? 1 : -1];

enum storage_kind { SYSTEM, PINNED, PINNED_WC, REGISTERED, MANAGED, FILE_MAPPING };

typedef struct {
    char message[512];
} failure_t;

typedef struct {
    enum storage_kind kind;
    int device;
    void *base;
    void *gpu;
    size_t extent;
    int64_t bytes;
    bool registered;
    bool locked;
    bool published;
} storage_t;

typedef struct {
    DLManagedTensor tensor;
    storage_t *storage;
    int64_t shape;
} export_t;

static pthread_mutex_t stats_mutex = PTHREAD_MUTEX_INITIALIZER;
static uint64_t live_bytes;
static uint64_t live_allocations;

static bool cuda_ok(cudaError_t status, const char *operation, failure_t *failure) {
    if (status == cudaSuccess) return true;
    snprintf(failure->message, sizeof(failure->message), "%s: %s",
             operation, cudaGetErrorString(status));
    return false;
}

static void system_error(failure_t *failure, const char *operation) {
    snprintf(failure->message, sizeof(failure->message), "%s: %s",
             operation, strerror(errno));
}

static bool require_attribute(enum cudaDeviceAttr key, int device,
                              const char *description, failure_t *failure) {
    int value;
    if (!cuda_ok(cudaDeviceGetAttribute(&value, key, device),
                 "cudaDeviceGetAttribute", failure)) return false;
    if (value) return true;
    snprintf(failure->message, sizeof(failure->message), "%s", description);
    return false;
}

static void release_storage(storage_t *storage) {
    if (!storage) return;
    if (storage->base) {
        int previous = storage->device;
        cudaGetDevice(&previous);
        cudaSetDevice(storage->device);
        // External DLPack storage is not tracked by Torch's stream allocator.
        if (storage->published) cudaDeviceSynchronize();
        if (storage->locked) munlock(storage->base, (size_t)storage->bytes);
        if (storage->registered) cudaHostUnregister(storage->base);
        if (storage->kind == PINNED || storage->kind == PINNED_WC)
            cudaFreeHost(storage->base);
        else if (storage->kind == MANAGED)
            cudaFree(storage->base);
        else
            munmap(storage->base, storage->extent);
        cudaSetDevice(previous);
    }
    pthread_mutex_lock(&stats_mutex);
    live_bytes -= storage->bytes;
    live_allocations--;
    pthread_mutex_unlock(&stats_mutex);
    free(storage);
}

static bool parse_kind(const char *name, enum storage_kind *kind, failure_t *failure) {
    static const struct {
        const char *name;
        enum storage_kind kind;
    } kinds[] = {
        {"system", SYSTEM}, {"pinned", PINNED}, {"pinned_wc", PINNED_WC},
        {"registered", REGISTERED}, {"managed", MANAGED}, {"file", FILE_MAPPING},
    };
    for (size_t i = 0; i < sizeof(kinds) / sizeof(kinds[0]); i++) {
        if (strcmp(name, kinds[i].name) == 0) {
            *kind = kinds[i].kind;
            return true;
        }
    }
    snprintf(failure->message, sizeof(failure->message), "unknown allocation kind: %s", name);
    return false;
}

static storage_t *read_storage(int fd, int64_t offset, int64_t bytes,
                               const char *name, int device, failure_t *failure) {
    storage_t *storage = NULL;
    enum storage_kind kind;
    struct stat status;
    int previous;
    bool success = false;

    if (offset < 0 || bytes < 0 || offset > INT64_MAX - bytes) {
        snprintf(failure->message, sizeof(failure->message), "invalid signed 64-bit file range");
        return NULL;
    }
    if (fd >= 0 && fstat(fd, &status) != 0) {
        system_error(failure, "fstat");
        return NULL;
    }
    if (fd >= 0 && (!S_ISREG(status.st_mode) || offset + bytes > status.st_size)) {
        snprintf(failure->message, sizeof(failure->message), "tensor range exceeds a regular file's size");
        return NULL;
    }
    if (!parse_kind(name, &kind, failure)) return NULL;
    if (!cuda_ok(cudaGetDevice(&previous), "cudaGetDevice", failure)) return NULL;
    if (!cuda_ok(cudaSetDevice(device), "cudaSetDevice", failure)) goto done;
    if ((kind == SYSTEM || kind == FILE_MAPPING) &&
        (!require_attribute(cudaDevAttrPageableMemoryAccess, device,
                            "system/file storage requires GPU access to pageable memory", failure) ||
         !require_attribute(cudaDevAttrPageableMemoryAccessUsesHostPageTables, device,
                            "system/file storage requires GPU host page tables", failure))) goto done;
    if (kind == MANAGED &&
        (!require_attribute(cudaDevAttrManagedMemory, device,
                            "managed memory is unsupported", failure) ||
         !require_attribute(cudaDevAttrConcurrentManagedAccess, device,
                            "managed storage requires concurrent managed access", failure))) goto done;
    if ((kind == PINNED || kind == PINNED_WC || kind == REGISTERED) &&
        !require_attribute(cudaDevAttrCanMapHostMemory, device,
                           "mapped host allocations are unsupported", failure)) goto done;
    if (kind == REGISTERED &&
        !require_attribute(cudaDevAttrHostRegisterSupported, device,
                           "CUDA host registration is unsupported", failure)) goto done;

    storage = calloc(1, sizeof(*storage));
    if (!storage) {
        snprintf(failure->message, sizeof(failure->message), "could not allocate storage owner");
        goto done;
    }
    storage->kind = kind;
    storage->device = device;
    storage->bytes = bytes;
    pthread_mutex_lock(&stats_mutex);
    live_bytes += bytes;
    live_allocations++;
    pthread_mutex_unlock(&stats_mutex);

    size_t length = bytes ? (size_t)bytes : 1;
    if (kind == PINNED || kind == PINNED_WC) {
        unsigned flags = cudaHostAllocMapped;
        if (kind == PINNED_WC) flags |= cudaHostAllocWriteCombined;
        if (!cuda_ok(cudaHostAlloc(&storage->base, length, flags), "cudaHostAlloc", failure)) goto done;
        if (!cuda_ok(cudaHostGetDevicePointer(&storage->gpu, storage->base, 0),
                     "cudaHostGetDevicePointer", failure)) goto done;
    } else if (kind == MANAGED) {
        if (!cuda_ok(cudaMallocManaged(&storage->base, length, cudaMemAttachGlobal),
                     "cudaMallocManaged", failure)) goto done;
        storage->gpu = storage->base;
    } else {
        size_t delta = 0;
        int flags = MAP_PRIVATE | MAP_ANONYMOUS;
        int map_fd = -1;
        off_t map_offset = 0;
        if (kind == FILE_MAPPING && bytes) {
            long page = sysconf(_SC_PAGESIZE);
            if (page <= 0) {
                snprintf(failure->message, sizeof(failure->message), "could not query host page size");
                goto done;
            }
            delta = (uint64_t)offset % (size_t)page;
            map_offset = offset - (int64_t)delta;
            map_fd = fd;
            flags = MAP_PRIVATE;
        }
        storage->extent = length + delta;
        void *mapped = mmap(NULL, storage->extent, PROT_READ | PROT_WRITE,
                            flags, map_fd, map_offset);
        if (mapped == MAP_FAILED) {
            system_error(failure, "mmap");
            goto done;
        }
        storage->base = mapped;
        storage->gpu = (char *)mapped + delta;
        if (kind == REGISTERED) {
            if (!cuda_ok(cudaHostRegister(mapped, length, cudaHostRegisterMapped),
                         "cudaHostRegister", failure)) goto done;
            storage->registered = true;
            if (!cuda_ok(cudaHostGetDevicePointer(&storage->gpu, mapped, 0),
                         "cudaHostGetDevicePointer", failure)) goto done;
        }
    }
    if (fd >= 0 && kind != FILE_MAPPING) {
        int64_t completed = 0;
        while (completed < bytes) {
            size_t chunk = bytes - completed < (8 << 20) ? (size_t)(bytes - completed) : (8 << 20);
            ssize_t count = pread(fd, (char *)storage->base + completed, chunk, offset + completed);
            if (count < 0 && errno == EINTR) continue;
            if (count < 0) {
                system_error(failure, "pread");
                goto done;
            }
            if (count == 0) {
                snprintf(failure->message, sizeof(failure->message), "unexpected EOF during tensor read");
                goto done;
            }
            completed += count;
        }
    }
    success = true;
done:
    if (!success) {
        release_storage(storage);
        storage = NULL;
    }
    cudaSetDevice(previous);
    return storage;
}

static void delete_tensor(DLManagedTensor *tensor) {
    export_t *exported = tensor->manager_ctx;
    release_storage(exported->storage);
    free(exported);
}

static void delete_capsule(PyObject *capsule) {
    if (PyCapsule_IsValid(capsule, "dltensor"))
        delete_tensor(PyCapsule_GetPointer(capsule, "dltensor"));
}

static PyObject *py_read(PyObject *self, PyObject *args) {
    (void)self;
    int fd, device;
    long long offset, bytes;
    const char *kind;
    if (!PyArg_ParseTuple(args, "iLLsi", &fd, &offset, &bytes, &kind, &device)) return NULL;
    if (fd < 0) return PyErr_Format(PyExc_ValueError, "invalid file descriptor");
    storage_t *storage;
    failure_t failure = {{0}};
    Py_BEGIN_ALLOW_THREADS
    storage = read_storage(fd, offset, bytes, kind, device, &failure);
    Py_END_ALLOW_THREADS
    if (!storage) return PyErr_Format(PyExc_RuntimeError, "%s", failure.message);
    export_t *exported = calloc(1, sizeof(*exported));
    if (!exported) {
        release_storage(storage);
        return PyErr_NoMemory();
    }
    exported->shape = bytes;
    exported->storage = storage;
    exported->tensor.dl_tensor = (DLTensor){
        .data = storage->gpu, .device = {kDLCUDA, device}, .ndim = 1,
        .dtype = {kDLUInt, 8, 1}, .shape = &exported->shape,
    };
    exported->tensor.manager_ctx = exported;
    exported->tensor.deleter = delete_tensor;
    PyObject *capsule = PyCapsule_New(&exported->tensor, "dltensor", delete_capsule);
    if (!capsule) {
        delete_tensor(&exported->tensor);
        return NULL;
    }
    storage->published = true;
    return capsule;
}

static PyObject *py_capabilities(PyObject *self, PyObject *args) {
    (void)self;
    int device;
    if (!PyArg_ParseTuple(args, "i", &device)) return NULL;
    static const struct {
        const char *name;
        enum cudaDeviceAttr attribute;
    } keys[] = {
        {"integrated", cudaDevAttrIntegrated},
        {"can_map_host_memory", cudaDevAttrCanMapHostMemory},
        {"managed_memory", cudaDevAttrManagedMemory},
        {"concurrent_managed_access", cudaDevAttrConcurrentManagedAccess},
        {"pageable_memory_access", cudaDevAttrPageableMemoryAccess},
        {"host_page_tables", cudaDevAttrPageableMemoryAccessUsesHostPageTables},
        {"host_register_supported", cudaDevAttrHostRegisterSupported},
        {"registered_host_pointer", cudaDevAttrCanUseHostPointerForRegisteredMem},
    };
    PyObject *result = PyDict_New();
    if (!result) return NULL;
    for (size_t i = 0; i < sizeof(keys) / sizeof(keys[0]); i++) {
        int value;
        failure_t failure;
        if (!cuda_ok(cudaDeviceGetAttribute(&value, keys[i].attribute, device),
                     "cudaDeviceGetAttribute", &failure)) {
            Py_DECREF(result);
            return PyErr_Format(PyExc_RuntimeError, "%s", failure.message);
        }
        PyObject *number = PyLong_FromLong(value);
        if (!number || PyDict_SetItemString(result, keys[i].name, number) != 0) {
            Py_XDECREF(number);
            Py_DECREF(result);
            return NULL;
        }
        Py_DECREF(number);
    }
    return result;
}

static PyObject *py_stats(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;
    pthread_mutex_lock(&stats_mutex);
    uint64_t bytes = live_bytes;
    uint64_t allocations = live_allocations;
    pthread_mutex_unlock(&stats_mutex);
    return Py_BuildValue("{s:K,s:K}", "live_bytes", (unsigned long long)bytes,
                         "live_allocations", (unsigned long long)allocations);
}

#include "_pool.c"
#include "_direct.c"
#include "_batch.c"

static PyMethodDef methods[] = {
    {"batch_executor", py_batch_executor, METH_VARARGS, NULL},
    {"batch_execute", py_batch_execute, METH_VARARGS, NULL},
    {"batch_stats", py_batch_stats, METH_O, NULL},
    {"direct_reader", py_direct_reader, METH_VARARGS, NULL},
    {"direct_bytes", py_direct_bytes, METH_VARARGS, NULL},
    {"direct_into", py_direct_into, METH_VARARGS, NULL},
    {"direct_bf16_into_fp32", py_direct_bf16_into_fp32, METH_VARARGS, NULL},
    {"direct_stats", py_direct_stats, METH_O, NULL},
    {"keep_allocator", py_keep_allocator, METH_O, NULL},
    {"pool_copy", py_pool_copy, METH_VARARGS, NULL},
    {"pool_contains", py_pool_contains, METH_VARARGS, NULL},
    {"read", py_read, METH_VARARGS, NULL},
    {"capabilities", py_capabilities, METH_VARARGS, NULL},
    {"storage_stats", py_stats, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL},
};
static PyModuleDef module = {
    PyModuleDef_HEAD_INIT, .m_name = "_b12x_loader_storage", .m_size = -1,
    .m_methods = methods,
};

PyMODINIT_FUNC PyInit__b12x_loader_storage(void) {
    PyObject *result = PyModule_Create(&module);
    if (result && PyModule_AddIntConstant(result, "ABI_VERSION", 1) < 0) {
        Py_DECREF(result);
        return NULL;
    }
    return result;
}
