#define _GNU_SOURCE  // 这个宏启用GNU扩展，包括pthread_barrier_t
#include <pthread.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>    // 添加math.h用于fabs
#include <string.h>  // 添加string.h用于memset
#include "native.h"
#include "optimization.h"
#include </home/kun/spack/opt/spack/linux-x86_64_v4/amdblis-5.2-7c4ickkpixrejjmg4j7x44csmdh4ucma/include/blis/blis.h>

void print_matrix(float *mat, int rows, int cols, const char* name) {
    printf("%s:\n", name);
    for (int i = 0; i < rows && i < 8; i++) {  // 最多打印8行
        for (int j = 0; j < cols && j < 8; j++) {  // 最多打印8列
            printf("%8.4f ", mat[i*cols + j]);
        }
        printf("\n");
    }
}

void test_sgemm(int size, int repeat) {
    int M, N, K;
    M = N = K = size;
    float alpha = 1.0f, beta = 0.0f;
    
    // Allocate memory
    float *A = (float*)aligned_alloc(32, M * K * sizeof(float));
    float *B = (float*)aligned_alloc(32, K * N * sizeof(float));
    float *C = (float*)aligned_alloc(32, M * N * sizeof(float));
    float *C_ref = (float*)malloc(M * N * sizeof(float));
    float *C_blas = (float*)malloc(M * N * sizeof(float));
    
    // Initialize data
    srand(42);
    for (int i = 0; i < M * K; i++) A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) B[i] = (float)rand() / RAND_MAX;

    memset(C, 0, M * N * sizeof(float));
    memset(C_ref, 0, M * N * sizeof(float));
    memset(C_blas, 0, M * N * sizeof(float));
    
    // the result of native version 
    Sgemm_native(M, N, K, alpha, A, K, B, N, beta, C_ref, N);
    
    // the result of optimization version
    Sgemm_latest(M, N, K, alpha, A, K, B, N, beta, C, N);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            M, N, K, alpha,
            A, K, B, N,
            beta,
            C_blas, N);
    // print_matrix(A, M, K, "Matrix A");
    // print_matrix(B, K, N, "Matrix B");
    // print_matrix(C, M, N, "My C");
    // print_matrix(C_blas, M, N, "BLAS C");
    
    
    double max_diff = 0.0;
    int id = -1;
    for (int i = 0; i < M * N; i++) {
        double diff = fabs(C[i] - C_ref[i]);
        if(diff > 1e-3) {
            max_diff = diff;
            id = i;
            break;}
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
        Sgemm_latest(M, N, K, alpha, A, K, B, N, beta, C, N);
    end = clock();
    double time_opt = (double)(end - start) / CLOCKS_PER_SEC / repeat;

    start = clock();
    for (int i = 0; i < repeat; i++)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, alpha,
                A, K, B, N,
                beta,
                C_blas, N);
    end = clock();
    double time_blas = (double)(end - start) / CLOCKS_PER_SEC / repeat;
    
    printf("M=%d, N=%d, K=%d\n", M, N, K);
    printf("Blas Sgemm cost :%.5f s GFLOPS = %.5f\n", time_blas,  (unsigned)(2 * M * N * K) / (time_blas * 1e9));
    printf("Native Sgemm cost :%.5f s GFLOPS = %.5f\n", time_basic, (unsigned)(2 * M * N * K) / (time_basic * 1e9));
    printf("Optimizated Sgemm cost :%.5f s GFLOPS = %.5f\n", time_opt, (unsigned)(2 * M * N * K) / (time_opt * 1e9));
    printf("speedup: %.2fx\n", time_basic / time_opt);
    printf("Reached %.2f%% of blas \n", time_blas/time_opt*100);
    printf("max error: %e\n", max_diff);
    printf("test %s\n\n", max_diff < 1e-2 ? "passed ✓" : "failed ✗");
    if(max_diff > 1e-2)
            printf("the first error at %d\n",id);
    
    free(A);
    free(B);
    free(C);
    free(C_ref);
    free(C_blas);
}