/* TEST-ONLY stand-in for <cuda.h>.
 *
 * This header exists so the CUDA adapter and the mock driver can be compiled
 * and exercised on machines without a CUDA toolkit. Type and struct names
 * follow the CUDA 13.4 Driver API reference for the checkpoint module
 * (docs.nvidia.com/cuda/cuda-driver-api, group CUDA_CHECKPOINT and the
 * CUcheckpoint* struct pages). Enum VALUES and struct LAYOUTS here are not
 * guaranteed to match the real header. A gpuckpt binary built with this
 * header reports "cuda-header: mock (UNVERIFIED)" and must never be pointed
 * at a real libcuda. On a CUDA host, build with CUDA_HOME set so the real
 * <cuda.h> is used instead.
 */
#ifndef GPUCKPT_MOCK_CUDA_H
#define GPUCKPT_MOCK_CUDA_H
#include <stddef.h>

typedef enum cudaError_enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_INVALID_DEVICE = 101,
    CUDA_ERROR_INVALID_IMAGE = 200,
    CUDA_ERROR_NOT_FOUND = 500,
    CUDA_ERROR_INVALID_HANDLE = 400,
    CUDA_ERROR_ILLEGAL_STATE = 401,
    CUDA_ERROR_ILLEGAL_ADDRESS = 700,
    CUDA_ERROR_NOT_SUPPORTED = 801,
    CUDA_ERROR_TIMEOUT = 909,
    CUDA_ERROR_UNKNOWN = 999
} CUresult;

typedef int CUdevice;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUctx_st *CUcontext;
typedef struct CUstream_st *CUstream;
typedef unsigned long long CUdeviceptr;
typedef unsigned long long cuuint64_t;
typedef struct CUuuid_st { char bytes[16]; } CUuuid;

typedef enum CUpointer_attribute_enum {
    CU_POINTER_ATTRIBUTE_CONTEXT = 1,
    CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2,
    CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3,
    CU_POINTER_ATTRIBUTE_HOST_POINTER = 4,
    CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9
} CUpointer_attribute;

#define CU_MEMHOSTALLOC_PORTABLE 0x01
#define CU_MEMHOSTREGISTER_PORTABLE 0x01
#define CU_STREAM_NON_BLOCKING 0x1

typedef enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
} CUdevice_attribute;

typedef enum CUprocessState_enum {
    CU_PROCESS_STATE_RUNNING = 0,
    CU_PROCESS_STATE_LOCKED,
    CU_PROCESS_STATE_CHECKPOINTED,
    CU_PROCESS_STATE_FAILED,
    CU_PROCESS_STATE_CHECKPOINTING,
    CU_PROCESS_STATE_RESTORING
} CUprocessState;

typedef struct CUcheckpointOperation_st *CUcheckpointOperationHandle;

typedef struct CUcheckpointLockArgs_st {
    unsigned int timeoutMs;     /* 0 = no timeout */
    unsigned int reserved0;
    cuuint64_t   reserved1[7];
} CUcheckpointLockArgs;

typedef struct CUcheckpointUnlockArgs_st {
    cuuint64_t reserved[8];
} CUcheckpointUnlockArgs;

typedef struct CUcheckpointCustomStoragePerDeviceData_st {
    CUdeviceptr devPtr;   /* zero-copy mapped device memory for the user to copy to/from */
    size_t      size;     /* size of mapped memory */
    CUstream    stream;   /* stream the driver synchronizes before completing */
} CUcheckpointCustomStoragePerDeviceData;

typedef struct CUcheckpointCustomStorageInfo_st {
    CUcheckpointOperationHandle             handle;         /* needed to complete the operation */
    CUcheckpointCustomStoragePerDeviceData *perDeviceData;  /* returned array; user sets NULL */
    unsigned int                            deviceCount;
} CUcheckpointCustomStorageInfo;

typedef struct CUcheckpointCheckpointArgs_st {
    CUcheckpointCustomStorageInfo **customStorageInfo_out;  /* NULL: checkpoint to host */
    char reserved[64 - sizeof(CUcheckpointCustomStorageInfo **)];
} CUcheckpointCheckpointArgs;

typedef struct CUcheckpointGpuPair_st {
    CUuuid oldGpu;
    CUuuid newGpu;
} CUcheckpointGpuPair;

typedef struct CUcheckpointRestoreArgs_st {
    CUcheckpointGpuPair            *gpuPairs;
    unsigned int                    gpuPairsCount;
    unsigned int                    padding0;
    CUcheckpointCustomStorageInfo **customStorageInfo_out;  /* NULL: restore from host */
    char reserved[64 - sizeof(CUcheckpointGpuPair *) - 2 * sizeof(unsigned int)
                     - sizeof(CUcheckpointCustomStorageInfo **)];
} CUcheckpointRestoreArgs;

#endif
