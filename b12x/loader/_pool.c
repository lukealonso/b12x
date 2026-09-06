/* Included by _storage.c. PyTorch owns suballocation and stream bookkeeping. */
typedef struct pool_segment {
    storage_t *storage;
    struct pool_segment *next;
} pool_segment_t;

static pthread_mutex_t pool_mutex = PTHREAD_MUTEX_INITIALIZER;
static pool_segment_t *pool_segments;
static pool_segment_t *last_segment;
/* MemPool keeps a raw backend pointer. Keep its Python owners through CUDA
   allocator shutdown; segment payloads still retire normally via pool_free. */
static PyObject *pool_allocators;

static PyObject *py_keep_allocator(PyObject *self, PyObject *allocator) {
    (void)self;
    if (!pool_allocators) pool_allocators = PyList_New(0);
    if (!pool_allocators || PyList_Append(pool_allocators, allocator) < 0) return NULL;
    Py_RETURN_NONE;
}

static void *pool_allocate(size_t bytes, int device, const char *kind) {
    if (bytes > INT64_MAX) return NULL;
    failure_t failure = {{0}};
    storage_t *storage = read_storage(-1, 0, (int64_t)bytes, kind, device, &failure);
    if (!storage) {
        fprintf(stderr, "b12x shared allocation failed: %s\n", failure.message);
        return NULL;
    }
    /* Host registration on ATS platforms need not page-lock system memory.
       Serving weights must stay resident even while file readahead competes. */
    if (mlock(storage->base, (size_t)storage->bytes) != 0) {
        fprintf(stderr, "b12x could not lock final weight storage: %s\n", strerror(errno));
        release_storage(storage);
        return NULL;
    }
    storage->locked = true;
    pool_segment_t *segment = malloc(sizeof(*segment));
    if (!segment) {
        release_storage(storage);
        return NULL;
    }
    segment->storage = storage;
    pthread_mutex_lock(&pool_mutex);
    segment->next = pool_segments;
    pool_segments = segment;
    pthread_mutex_unlock(&pool_mutex);
    return storage->gpu;
}

void *b12x_registered_alloc(size_t bytes, int device, cudaStream_t stream) {
    (void)stream;
    return pool_allocate(bytes, device, "registered");
}

void *b12x_pinned_alloc(size_t bytes, int device, cudaStream_t stream) {
    (void)stream;
    return pool_allocate(bytes, device, "pinned");
}

void *b12x_pinned_wc_alloc(size_t bytes, int device, cudaStream_t stream) {
    (void)stream;
    return pool_allocate(bytes, device, "pinned_wc");
}

void *b12x_managed_alloc(size_t bytes, int device, cudaStream_t stream) {
    (void)stream;
    return pool_allocate(bytes, device, "managed");
}

void b12x_pool_free(void *pointer, size_t bytes, int device, cudaStream_t stream) {
    (void)bytes;
    (void)device;
    (void)stream;
    pthread_mutex_lock(&pool_mutex);
    pool_segment_t **link = &pool_segments;
    while (*link && (*link)->storage->gpu != pointer) link = &(*link)->next;
    pool_segment_t *segment = *link;
    if (segment) {
        *link = segment->next;
        if (last_segment == segment) last_segment = NULL;
    }
    pthread_mutex_unlock(&pool_mutex);
    if (segment) {
        release_storage(segment->storage);
        free(segment);
    }
}

static bool contains(storage_t *storage, uintptr_t pointer, uint64_t bytes) {
    uintptr_t base = (uintptr_t)storage->gpu;
    return pointer >= base && pointer - base <= (uint64_t)storage->bytes &&
           bytes <= (uint64_t)storage->bytes - (pointer - base);
}

/* Caller holds pool_mutex. */
static storage_t *find_segment(uintptr_t pointer, uint64_t bytes) {
    if (last_segment && contains(last_segment->storage, pointer, bytes))
        return last_segment->storage;
    for (pool_segment_t *segment = pool_segments; segment; segment = segment->next) {
        if (contains(segment->storage, pointer, bytes)) {
            last_segment = segment;
            return segment->storage;
        }
    }
    return NULL;
}

static PyObject *py_pool_contains(PyObject *self, PyObject *args) {
    (void)self;
    unsigned long long pointer, bytes;
    if (!PyArg_ParseTuple(args, "KK", &pointer, &bytes)) return NULL;
    pthread_mutex_lock(&pool_mutex);
    bool found = find_segment(pointer, bytes) != NULL;
    pthread_mutex_unlock(&pool_mutex);
    return PyBool_FromLong(found);
}

/* Inputs are validated Tensor pointers, kept alive by the Python caller. */
static PyObject *py_pool_copy(PyObject *self, PyObject *args) {
    (void)self;
    unsigned long long destination, source, bytes, stream;
    if (!PyArg_ParseTuple(args, "KKKK", &destination, &source, &bytes, &stream)) return NULL;
    failure_t failure = {{0}};
    bool found = false, success = true;
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&pool_mutex);
    storage_t *storage = find_segment(destination, bytes);
    if (storage) {
        found = true;
        success = cuda_ok(cudaStreamSynchronize((cudaStream_t)(uintptr_t)stream),
                          "synchronize before host weight write", &failure);
        if (success && bytes) {
            size_t delta = destination - (uintptr_t)storage->gpu;
            memcpy((char *)storage->base + delta, (void *)(uintptr_t)source, bytes);
        }
    }
    pthread_mutex_unlock(&pool_mutex);
    Py_END_ALLOW_THREADS
    if (!success) return PyErr_Format(PyExc_RuntimeError, "%s", failure.message);
    return PyBool_FromLong(found);
}
