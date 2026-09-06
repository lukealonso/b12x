/* Read-only O_DIRECT ceiling test with fixed, locked buffers. */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define BLOCK_BYTES (8 << 20)
static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
static char **paths;
static int path_count, next_path, failed;
static uint64_t total_bytes, total_reads;

static void *read_files(void *unused) {
    (void)unused;
    void *buffer = NULL;
    if (posix_memalign(&buffer, 4096, BLOCK_BYTES) || mlock(buffer, BLOCK_BYTES)) {
        perror("allocate/lock I/O buffer");
        exit(1);
    }
    uint64_t bytes = 0, reads = 0;
    int error = 0;
    for (;;) {
        pthread_mutex_lock(&mutex);
        int index = next_path++;
        pthread_mutex_unlock(&mutex);
        if (index >= path_count) break;
        int fd = open(paths[index], O_RDONLY | O_DIRECT | O_CLOEXEC);
        struct stat status;
        if (fd < 0 || fstat(fd, &status) || !S_ISREG(status.st_mode)) {
            perror(paths[index]);
            if (fd >= 0) close(fd);
            error = 1;
            break;
        }
        off_t offset = 0;
        while (offset < status.st_size) {
            ssize_t count = pread(fd, buffer, BLOCK_BYTES, offset);
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0 || (count % 4096 && offset + count != status.st_size)) {
                perror("O_DIRECT read");
                error = 1;
                break;
            }
            offset += count;
            bytes += count;
            reads++;
        }
        close(fd);
        if (error) break;
    }
    munlock(buffer, BLOCK_BYTES);
    free(buffer);
    pthread_mutex_lock(&mutex);
    total_bytes += bytes;
    total_reads += reads;
    failed |= error;
    pthread_mutex_unlock(&mutex);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s WORKERS FILE...\n", argv[0]);
        return 2;
    }
    char *end;
    long workers = strtol(argv[1], &end, 10);
    if (*end || workers < 1 || workers > 32) return 2;
    paths = argv + 2;
    path_count = argc - 2;
    pthread_t threads[32];
    struct timespec start, stop;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (long i = 0; i < workers; i++) {
        if (pthread_create(&threads[i], NULL, read_files, NULL)) return 1;
    }
    for (long i = 0; i < workers; i++) pthread_join(threads[i], NULL);
    clock_gettime(CLOCK_MONOTONIC, &stop);
    double seconds = stop.tv_sec - start.tv_sec + (stop.tv_nsec - start.tv_nsec) * 1e-9;
    printf("{\"workers\":%ld,\"buffer_bytes\":%ld,\"bytes\":%llu,"
           "\"reads\":%llu,\"seconds\":%.6f,\"GB_per_second\":%.6f,\"failed\":%d}\n",
           workers, workers * BLOCK_BYTES, (unsigned long long)total_bytes,
           (unsigned long long)total_reads, seconds, total_bytes / seconds / 1e9, failed);
    return failed;
}
