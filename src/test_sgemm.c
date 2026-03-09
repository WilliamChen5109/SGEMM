#include <time.h>
#include <stdlib.h>
#include "native.h"

void test_sgemm(int size, int repeat) {
    int M, N, K;
    M = N = K = size;
    float alpha = 1.0f, beta = 0.0f;
    
    // Allocate memory
    float *A = (float*)aligned_alloc(32, M * K * sizeof(float));
    float *B = (float*)aligned_alloc(32, K * N * sizeof(float));
    float *C = (float*)aligned_alloc(32, M * N * sizeof(float));
    float *C_ref = (float*)malloc(M * N * sizeof(float));
    
    // Initialize data
    srand(42);
    for (int i = 0; i < M * K; i++) A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) B[i] = (float)rand() / RAND_MAX;

    memset(C, 0, M * N * sizeof(float));
    memset(C_ref, 0, M * N * sizeof(float));
    
    // the result of native version 
    Sgemm_native(M, N, K, alpha, A, K, B, N, beta, C_ref, N);
    
    // the result of optimization version
    Sgemm_native(M, N, K, alpha, A, K, B, N, beta, C, N);
    
    
    double max_diff = 0.0;
    for (int i = 0; i < M * N; i++) {
        double diff = fabs(C[i] - C_ref[i]);
        if (diff > max_diff) max_diff = diff;
    }

    // run time test
    clock_t start = clock();
    for (int i = 0; i < repeat; i++)
        Sgemm_native(M, N, K, alpha, A, K, B, N, beta, C_ref, N);
    clock_t end = clock();
    double time_basic = (double)(end - start) / CLOCKS_PER_SEC / repeat;

    start = clock();
    for (int i = 0; i < repeat; i++)
        Sgemm_native(M, N, K, alpha, A, K, B, N, beta, C, N);
    end = clock();
    double time_opt = (double)(end - start) / CLOCKS_PER_SEC / repeat;
    
    printf("M=%d, N=%d, K=%d\n", M, N, K);
    printf("Native Sgemm cost: %.3f s\n", time_basic);
    printf("Optimizated Sgemm cost: %.3f s\n", time_opt);
    printf("speedup: %.2fx\n", time_basic / time_opt);
    printf("max error: %e\n", max_diff);
    printf("test %s\n\n", max_diff < 1e-4 ? "passed ✓" : "failed ✗");
    
    free(A);
    free(B);
    free(C);
    free(C_ref);
}