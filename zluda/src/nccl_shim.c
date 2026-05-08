#include <stdint.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

typedef int ncclResult_t;
typedef void *cudaStream_t;

typedef struct {
    char internal[128];
} ncclUniqueId;

typedef struct HetGpuNcclComm {
    int nranks;
    int rank;
    int device;
    int aborted;
    uint64_t sequence;
} *ncclComm_t;

enum {
    ncclSuccess = 0,
    ncclUnhandledCudaError = 1,
    ncclSystemError = 2,
    ncclInvalidArgument = 3,
    ncclInvalidUsage = 5,
};

static int hetgpu_nccl_logs_enabled(void) {
    const char *env = getenv("HETGPU_NCCL_LOGS");
    return env && (strcmp(env, "1") == 0 || strcmp(env, "true") == 0 || strcmp(env, "on") == 0);
}

#define NCCL_LOG(...) do { if (hetgpu_nccl_logs_enabled()) fprintf(stderr, __VA_ARGS__); } while (0)

extern int hetgpu_pacc_nccl_all_reduce_f32(
    const float *sendbuff,
    float *recvbuff,
    size_t count,
    int op,
    int rank,
    int nranks
) __attribute__((weak));

extern int hetgpu_pacc_nccl_reduce_sum_f32(
    const float *rank_inputs,
    float *recvbuff,
    size_t count,
    int nranks
) __attribute__((weak));

static int env_enabled(const char *name) {
    const char *env = getenv(name);
    return env && (strcmp(env, "1") == 0 || strcmp(env, "true") == 0 || strcmp(env, "on") == 0);
}

static int read_env_i32(const char *name, int fallback) {
    const char *env = getenv(name);
    if (!env || !*env) return fallback;
    char *end = NULL;
    long v = strtol(env, &end, 10);
    if (!end || *end != '\0') return fallback;
    return (int)v;
}

static size_t dtype_size(int datatype) {
    switch (datatype) {
        case 0: return 1;  /* int8/char */
        case 1: return 1;  /* uint8 */
        case 2: return 4;  /* int32 */
        case 3: return 4;  /* uint32 */
        case 4: return 8;  /* int64 */
        case 5: return 8;  /* uint64 */
        case 6: return 2;  /* float16 */
        case 7: return 4;  /* float32 */
        case 8: return 8;  /* float64 */
        case 9: return 2;  /* bfloat16 */
        default: return 0;
    }
}

const char *ncclGetErrorString(ncclResult_t result) {
    switch (result) {
        case ncclSuccess: return "ncclSuccess";
        case ncclUnhandledCudaError: return "ncclUnhandledCudaError";
        case ncclSystemError: return "ncclSystemError";
        case ncclInvalidArgument: return "ncclInvalidArgument";
        case ncclInvalidUsage: return "ncclInvalidUsage";
        default: return "ncclUnknownError";
    }
}

const char *ncclGetLastError(ncclComm_t comm) {
    (void)comm;
    return "hetGPU NCCL shim: no async error";
}

ncclResult_t ncclGetVersion(int *version) {
    if (!version) return ncclInvalidArgument;
    *version = 21200;
    return ncclSuccess;
}

ncclResult_t ncclGetUniqueId(ncclUniqueId *uniqueId) {
    if (!uniqueId) return ncclInvalidArgument;
    memset(uniqueId, 0x50, sizeof(*uniqueId));
    memcpy(uniqueId->internal, "HETGPU-PACC-NCCL-SHIM", 21);
    return ncclSuccess;
}

static ncclResult_t make_comm(ncclComm_t *comm, int nranks, int rank, int device) {
    if (!comm || nranks <= 0 || rank < 0 || rank >= nranks) return ncclInvalidArgument;
    struct HetGpuNcclComm *created = (struct HetGpuNcclComm *)calloc(1, sizeof(*created));
    if (!created) return ncclSystemError;
    created->nranks = nranks;
    created->rank = rank;
    created->device = device;
    created->sequence = 0;
    *comm = created;
    return ncclSuccess;
}

ncclResult_t ncclCommInitRank(ncclComm_t *comm, int nranks, ncclUniqueId commId, int rank) {
    (void)commId;
    if (nranks <= 0) nranks = read_env_i32("WORLD_SIZE", 1);
    if (rank < 0) rank = read_env_i32("RANK", 0);
    NCCL_LOG("[hetGPU nccl_shim] ncclCommInitRank nranks=%d rank=%d\n", nranks, rank);
    return make_comm(comm, nranks, rank, rank);
}

ncclResult_t ncclCommInitAll(ncclComm_t *comms, int ndev, const int *devlist) {
    if (!comms || ndev <= 0) return ncclInvalidArgument;
    NCCL_LOG("[hetGPU nccl_shim] ncclCommInitAll ndev=%d\n", ndev);
    for (int i = 0; i < ndev; ++i) {
        int device = devlist ? devlist[i] : i;
        ncclResult_t r = make_comm(&comms[i], ndev, i, device);
        if (r != ncclSuccess) return r;
    }
    return ncclSuccess;
}

ncclResult_t ncclCommDestroy(ncclComm_t comm) {
    free(comm);
    return ncclSuccess;
}

ncclResult_t ncclCommAbort(ncclComm_t comm) {
    if (comm) comm->aborted = 1;
    free(comm);
    return ncclSuccess;
}

ncclResult_t ncclCommFinalize(ncclComm_t comm) {
    (void)comm;
    return ncclSuccess;
}

ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError) {
    if (!asyncError) return ncclInvalidArgument;
    *asyncError = (comm && comm->aborted) ? ncclSystemError : ncclSuccess;
    return ncclSuccess;
}

ncclResult_t ncclCommCount(const ncclComm_t comm, int *count) {
    if (!comm || !count) return ncclInvalidArgument;
    *count = comm->nranks;
    return ncclSuccess;
}

ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int *device) {
    if (!comm || !device) return ncclInvalidArgument;
    *device = comm->device;
    return ncclSuccess;
}

ncclResult_t ncclCommUserRank(const ncclComm_t comm, int *rank) {
    if (!comm || !rank) return ncclInvalidArgument;
    *rank = comm->rank;
    return ncclSuccess;
}

ncclResult_t ncclGroupStart(void) {
    return ncclSuccess;
}

ncclResult_t ncclGroupEnd(void) {
    return ncclSuccess;
}

static int ensure_dir(const char *path) {
    if (mkdir(path, 0777) == 0 || errno == EEXIST) return 0;
    return -1;
}

static int wait_for_file(const char *path, int timeout_ms) {
    struct timespec sleep_time;
    sleep_time.tv_sec = 0;
    sleep_time.tv_nsec = 10 * 1000 * 1000;
    int waited = 0;
    while (access(path, R_OK) != 0) {
        if (waited >= timeout_ms) return -1;
        nanosleep(&sleep_time, NULL);
        waited += 10;
    }
    return 0;
}

static int write_file_exact(const char *path, const void *data, size_t bytes) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    size_t written = fwrite(data, 1, bytes, f);
    int close_rc = fclose(f);
    return (written == bytes && close_rc == 0) ? 0 : -1;
}

static int read_file_exact(const char *path, void *data, size_t bytes) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    size_t got = fread(data, 1, bytes, f);
    int close_rc = fclose(f);
    return (got == bytes && close_rc == 0) ? 0 : -1;
}

static void publish_abort(const char *abort_path, const char *reason) {
    if (!abort_path || !*abort_path) return;
    if (!reason || !*reason) reason = "aborted\n";
    (void)write_file_exact(abort_path, reason, strlen(reason));
}

static int wait_for_result_or_abort(const char *result_path, const char *abort_path, int timeout_ms) {
    struct timespec sleep_time;
    sleep_time.tv_sec = 0;
    sleep_time.tv_nsec = 10 * 1000 * 1000;
    int waited = 0;
    while (access(result_path, R_OK) != 0) {
        if (abort_path && access(abort_path, R_OK) == 0) return 1;
        if (waited >= timeout_ms) return -1;
        nanosleep(&sleep_time, NULL);
        waited += 10;
    }
    return 0;
}

static ncclResult_t rendezvous_allreduce_f32(
    const float *sendbuff,
    float *recvbuff,
    size_t count,
    int op,
    ncclComm_t comm
) {
    if (!comm || op != 0) return ncclInvalidArgument;

    int timeout_ms = read_env_i32("HETGPU_NCCL_TIMEOUT_MS", 60000);
    const char *job = getenv("HETGPU_NCCL_COMM_ID");
    if (!job || !*job) job = getenv("MASTER_PORT");
    if (!job || !*job) job = "default";

    char root[256];
    char call_dir[320];
    char rank_path[384];
    char result_path[384];
    char abort_path[384];
    snprintf(root, sizeof(root), "/dev/shm/hetgpu_nccl_%s", job);
    snprintf(call_dir, sizeof(call_dir), "%s/%llu", root, (unsigned long long)comm->sequence++);
    snprintf(rank_path, sizeof(rank_path), "%s/rank_%d.bin", call_dir, comm->rank);
    snprintf(result_path, sizeof(result_path), "%s/result.bin", call_dir);
    snprintf(abort_path, sizeof(abort_path), "%s/abort", call_dir);

    if (ensure_dir(root) != 0 || ensure_dir(call_dir) != 0) {
        perror("[hetGPU nccl_shim] mkdir");
        return ncclSystemError;
    }
    if (write_file_exact(rank_path, sendbuff, count * sizeof(float)) != 0) {
        perror("[hetGPU nccl_shim] write rank file");
        return ncclSystemError;
    }

    if (comm->rank == 0) {
        int use_pacc = env_enabled("HETGPU_NCCL_USE_PACC");
        float *accum = (float *)calloc(count, sizeof(float));
        float *rank_inputs = NULL;
        float *tmp = NULL;
        if (use_pacc) {
            rank_inputs = (float *)malloc(count * sizeof(float) * (size_t)comm->nranks);
        } else {
            tmp = (float *)malloc(count * sizeof(float));
        }
        if (!accum || (use_pacc && !rank_inputs) || (!use_pacc && !tmp)) {
            publish_abort(abort_path, "rank0 allocation failed\n");
            free(accum);
            free(rank_inputs);
            free(tmp);
            return ncclSystemError;
        }

        for (int r = 0; r < comm->nranks; ++r) {
            char path[384];
            float *dst = rank_inputs ? rank_inputs + ((size_t)r * count) : tmp;
            snprintf(path, sizeof(path), "%s/rank_%d.bin", call_dir, r);
            if (wait_for_file(path, timeout_ms) != 0 || read_file_exact(path, dst, count * sizeof(float)) != 0) {
                fprintf(stderr, "[hetGPU nccl_shim] timed out reading rank %d for allreduce\n", r);
                publish_abort(abort_path, "rank input timeout\n");
                free(accum);
                free(rank_inputs);
                free(tmp);
                return ncclSystemError;
            }
            if (!rank_inputs) {
                for (size_t i = 0; i < count; ++i) accum[i] += dst[i];
            }
        }

        if (use_pacc) {
            if (hetgpu_pacc_nccl_reduce_sum_f32) {
                int rc = hetgpu_pacc_nccl_reduce_sum_f32(
                    rank_inputs,
                    accum,
                    count,
                    comm->nranks
                );
                if (rc != 0) {
                    if (env_enabled("HETGPU_NCCL_CPU_FALLBACK_AFTER_PACC")) {
                        fprintf(stderr, "[hetGPU nccl_shim] PACC reduce-sum hook failed rc=%d; falling back to CPU sum\n", rc);
                        for (int r = 0; r < comm->nranks; ++r) {
                            float *src = rank_inputs + ((size_t)r * count);
                            for (size_t i = 0; i < count; ++i) accum[i] += src[i];
                        }
                    } else {
                        fprintf(stderr, "[hetGPU nccl_shim] PACC reduce-sum hook failed rc=%d\n", rc);
                        publish_abort(abort_path, "PACC reduce-sum hook failed\n");
                        free(accum);
                        free(rank_inputs);
                        free(tmp);
                        return ncclSystemError;
                    }
                }
            } else if (hetgpu_pacc_nccl_all_reduce_f32) {
                for (int r = 0; r < comm->nranks; ++r) {
                    float *src = rank_inputs + ((size_t)r * count);
                    for (size_t i = 0; i < count; ++i) accum[i] += src[i];
                }
                int rc = hetgpu_pacc_nccl_all_reduce_f32(accum, accum, count, op, comm->rank, comm->nranks);
                if (rc != 0) {
                    if (env_enabled("HETGPU_NCCL_CPU_FALLBACK_AFTER_PACC")) {
                        fprintf(stderr, "[hetGPU nccl_shim] PACC allreduce hook failed rc=%d; keeping CPU sum result\n", rc);
                    } else {
                        fprintf(stderr, "[hetGPU nccl_shim] PACC allreduce hook failed rc=%d\n", rc);
                        publish_abort(abort_path, "PACC allreduce hook failed\n");
                        free(accum);
                        free(rank_inputs);
                        free(tmp);
                        return ncclSystemError;
                    }
                }
            } else {
                fprintf(stderr, "[hetGPU nccl_shim] PACC hook symbol unavailable\n");
                publish_abort(abort_path, "PACC hook symbol unavailable\n");
                free(accum);
                free(rank_inputs);
                free(tmp);
                return ncclSystemError;
            }
        }

        if (write_file_exact(result_path, accum, count * sizeof(float)) != 0) {
            perror("[hetGPU nccl_shim] write result");
            publish_abort(abort_path, "result write failed\n");
            free(accum);
            free(rank_inputs);
            free(tmp);
            return ncclSystemError;
        }
        free(accum);
        free(rank_inputs);
        free(tmp);
    }

    int wait_rc = wait_for_result_or_abort(result_path, abort_path, timeout_ms);
    if (wait_rc == 1) {
        fprintf(stderr, "[hetGPU nccl_shim] allreduce aborted by rank 0\n");
        return ncclSystemError;
    }
    if (wait_rc != 0 || read_file_exact(result_path, recvbuff, count * sizeof(float)) != 0) {
        fprintf(stderr, "[hetGPU nccl_shim] timed out reading allreduce result\n");
        return ncclSystemError;
    }

    return ncclSuccess;
}

ncclResult_t ncclAllReduce(
    const void *sendbuff,
    void *recvbuff,
    size_t count,
    int datatype,
    int op,
    ncclComm_t comm,
    cudaStream_t stream
) {
    (void)op;
    (void)comm;
    (void)stream;
    size_t item_size = dtype_size(datatype);
    if (!sendbuff || !recvbuff || item_size == 0) return ncclInvalidArgument;
    NCCL_LOG("[hetGPU nccl_shim] ncclAllReduce count=%zu datatype=%d bytes=%zu\n",
             count, datatype, count * item_size);

    if (datatype == 7 && op == 0 && comm && comm->nranks > 1) {
        return rendezvous_allreduce_f32((const float *)sendbuff, (float *)recvbuff, count, op, comm);
    }

    if (datatype == 7 && op == 0 && env_enabled("HETGPU_NCCL_USE_PACC") && hetgpu_pacc_nccl_all_reduce_f32) {
        int rank = comm ? comm->rank : read_env_i32("RANK", 0);
        int nranks = comm ? comm->nranks : read_env_i32("WORLD_SIZE", 1);
        int rc = hetgpu_pacc_nccl_all_reduce_f32(
            (const float *)sendbuff,
            (float *)recvbuff,
            count,
            op,
            rank,
            nranks
        );
        return rc == 0 ? ncclSuccess : ncclSystemError;
    }

    if (sendbuff != recvbuff) memcpy(recvbuff, sendbuff, count * item_size);
    return ncclSuccess;
}

ncclResult_t ncclBroadcast(
    const void *sendbuff,
    void *recvbuff,
    size_t count,
    int datatype,
    int root,
    ncclComm_t comm,
    cudaStream_t stream
) {
    (void)root;
    return ncclAllReduce(sendbuff, recvbuff, count, datatype, 0, comm, stream);
}

ncclResult_t ncclReduce(
    const void *sendbuff,
    void *recvbuff,
    size_t count,
    int datatype,
    int op,
    int root,
    ncclComm_t comm,
    cudaStream_t stream
) {
    (void)root;
    return ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
}

ncclResult_t ncclAllGather(
    const void *sendbuff,
    void *recvbuff,
    size_t sendcount,
    int datatype,
    ncclComm_t comm,
    cudaStream_t stream
) {
    return ncclAllReduce(sendbuff, recvbuff, sendcount, datatype, 0, comm, stream);
}

ncclResult_t ncclReduceScatter(
    const void *sendbuff,
    void *recvbuff,
    size_t recvcount,
    int datatype,
    int op,
    ncclComm_t comm,
    cudaStream_t stream
) {
    return ncclAllReduce(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
}

ncclResult_t ncclSend(const void *sendbuff, size_t count, int datatype, int peer, ncclComm_t comm, cudaStream_t stream) {
    (void)sendbuff;
    (void)count;
    (void)datatype;
    (void)peer;
    (void)comm;
    (void)stream;
    return ncclSuccess;
}

ncclResult_t ncclRecv(void *recvbuff, size_t count, int datatype, int peer, ncclComm_t comm, cudaStream_t stream) {
    (void)recvbuff;
    (void)count;
    (void)datatype;
    (void)peer;
    (void)comm;
    (void)stream;
    return ncclSuccess;
}
