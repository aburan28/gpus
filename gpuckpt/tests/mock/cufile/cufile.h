/* TEST-ONLY stand-in for <cufile.h> (GPUDirect Storage), written from the
 * cuFile API reference (docs.nvidia.com/gpudirect-storage/api-reference-guide).
 * Used when the CUDA toolkit's cufile.h is absent; a binary built with it
 * reports "cufile-header: mock (UNVERIFIED)" and must not be pointed at a
 * real libcufile. */
#ifndef GPUCKPT_MOCK_CUFILE_H
#define GPUCKPT_MOCK_CUFILE_H
#include <cuda.h>
#include <sys/types.h>

#define CUFILEOP_BASE_ERR 5000
typedef enum CUfileOpError {
    CU_FILE_SUCCESS = 0,
    CU_FILE_DRIVER_NOT_INITIALIZED = 5001,
    CU_FILE_DRIVER_INVALID_PROPS = 5002,
    CU_FILE_DRIVER_UNSUPPORTED_LIMIT = 5003,
    CU_FILE_DRIVER_VERSION_MISMATCH = 5004,
    CU_FILE_DRIVER_VERSION_READ_ERROR = 5005,
    CU_FILE_DRIVER_CLOSING = 5006,
    CU_FILE_PLATFORM_NOT_SUPPORTED = 5007,
    CU_FILE_IO_NOT_SUPPORTED = 5008,
    CU_FILE_DEVICE_NOT_SUPPORTED = 5009
} CUfileOpError;

typedef struct CUfileError {
    CUfileOpError err;
    CUresult      cu_err;
} CUfileError_t;

typedef enum CUfileFileHandleType {
    CU_FILE_HANDLE_TYPE_OPAQUE_FD = 1,
    CU_FILE_HANDLE_TYPE_OPAQUE_WIN32 = 2,
    CU_FILE_HANDLE_TYPE_USERSPACE_FS = 3
} CUfileFileHandleType;

typedef struct CUfileFSOps CUfileFSOps_t;

typedef struct CUfileDescr_t {
    CUfileFileHandleType type;
    union { int fd; void *handle; } handle;
    const CUfileFSOps_t *fs_ops;
} CUfileDescr_t;

typedef void *CUfileHandle_t;
#endif
