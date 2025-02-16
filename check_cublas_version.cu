#include <cstdio>
#include <cublas_v2.h>

int main() {
    cublasHandle_t handle;
    // Create a cuBLAS handle
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
    
    int version = 0;
    // Get the version of cuBLAS
    if (cublasGetVersion_v2(handle, &version) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "Failed to get cuBLAS version\n");
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    
    printf("cuBLAS version: %d\n", version);
    cublasDestroy(handle);
    return EXIT_SUCCESS;
}
