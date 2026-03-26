#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>
#include <stdint.h>
/*
==========v1========
循环重排
i-j-k-->i-k-j
从而减少cache miss
====================
*/
void Sgemm_v1(int M, int N, int K, 
                 float alpha, 
                 const float *A, int lda,
                 const float *B, int ldb,
                 float beta,
                 float *C, int ldc){
    
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float aik = A[i * lda + k];
            for (int j = 0; j < N; j++) {
                C[i*ldc + j] += alpha * aik * B[k*ldb + j] + beta * C[i*ldc + j];
            }
        }
    }
}

void transpose(const float *src, float *dst, int rows, int cols) {

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            dst[c*rows + r] = src[r*cols + c]; 
        }
    }

}

/*
==========v2==========
transpose B->BT
======================
*/
void Sgemm_v2(int M, int N, int K, 
              float alpha, 
              const float *A, int lda,
              const float *B, int ldb,
              float beta,
              float *C, int ldc) {
    
    float *BT = (float*)aligned_alloc(32, K * N * sizeof(float));
    transpose(B,BT,K,N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * lda + k] * BT[j * K + k];
            }
            C[i * ldc + j] += alpha * sum;
        }
    }
}

void AddDot1x4(int K, float alpha, float beta,
                  const float *A, int lda,
                  const float *B, int ldb,
                  float *C, int ldc) {
    
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    
    for (int k = 0; k < K; k++) {
        c0 += A[k] * B[k];
        c1 += A[k] * B[k + ldb];
        c2 += A[k] * B[k + 2 * ldb];
        c3 += A[k] * B[k + 3 * ldb];
    }
    
    C[0] = alpha * c0 + beta * C[0];
    C[1] = alpha * c1 + beta * C[1];
    C[2] = alpha * c2 + beta * C[2];
    C[3] = alpha * c3 + beta * C[3];
}

void AddDot1x4_reg(int K, float alpha, float beta,
                  const float *A, int lda,
                  const float *B, int ldb,
                  float *C, int ldc) {
    
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;

    const float *a0p, *b0p, *b1p, *b2p, *b3p;
    a0p = &A[0];
    b0p = &B[0];
    b1p = &B[ldb];
    b2p = &B[2 * ldb];
    b3p = &B[3 * ldb];
    
    for (int k = 0; k < K; k++) {
        c0 += *a0p * *b0p++;
        c1 += *a0p * *b1p++;
        c2 += *a0p * *b2p++;
        c3 += *a0p++ * *b3p++;
    }
    
    C[0] = alpha * c0 + beta * C[0];
    C[1] = alpha * c1 + beta * C[1];
    C[2] = alpha * c2 + beta * C[2];
    C[3] = alpha * c3 + beta * C[3];
}
/*
==========v3==========
1x4 block kernel
======================
*/

void Sgemm_v3(int M, int N, int K, 
              float alpha, 
              const float *A, int lda,
              const float *B, int ldb,
              float beta,
              float *C, int ldc) {
    
    float *BT = (float*)aligned_alloc(32, K * N * sizeof(float));
    transpose(B,BT,K,N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j+=4) {
            AddDot1x4_reg(K, alpha, beta,
                        &A[i * lda], lda, 
                        &BT[j * K], K, 
                        &C[i * ldc + j], ldc);
        }
    }
    free(BT);
}

void AddDot4x4(int K, float alpha, float beta,
                  const float *A, int lda,
                  const float *B, int ldb,
                  float *C, int ldc) {
    
    float c00 = 0.0f, c01 = 0.0f, c02 = 0.0f, c03 = 0.0f,
            c10 = 0.0f, c11 = 0.0f, c12 = 0.0f, c13 = 0.0f,
            c20 = 0.0f, c21 = 0.0f, c22 = 0.0f, c23 = 0.0f,
            c30 = 0.0f, c31 = 0.0f, c32 = 0.0f, c33 = 0.0f;
    
    for (int k = 0; k < K; k++) {
        c00 += A[k] * B[k];
        c01 += A[k] * B[k + ldb];
        c02 += A[k] * B[k + ldb * 2];
        c03 += A[k] * B[k + ldb * 3];

        c10 += A[lda + k] * B[k];
        c11 += A[lda + k] * B[k + ldb * 1];
        c12 += A[lda + k] * B[k + ldb * 2];
        c13 += A[lda + k] * B[k + ldb * 3];

        c20 += A[2 * lda + k] * B[k];
        c21 += A[2 * lda + k] * B[k + ldb * 1];
        c22 += A[2 * lda + k] * B[k + ldb * 2];
        c23 += A[2 * lda + k] * B[k + ldb * 3];

        c30 += A[3 * lda + k] * B[k];
        c31 += A[3 * lda + k] * B[k + ldb * 1];
        c32 += A[3 * lda + k] * B[k + ldb * 2];
        c33 += A[3 * lda + k] * B[k + ldb * 3];
    }
    
        C[0] = alpha * c00;
        C[1] = alpha * c01;
        C[2] = alpha * c02;
        C[3] = alpha * c03;
        
        C[ldc] = alpha * c10;
        C[ldc + 1] = alpha * c11;
        C[ldc + 2] = alpha * c12;
        C[ldc + 3] = alpha * c13;
        
        C[2 * ldc] += alpha * c20;
        C[2 * ldc + 1] = alpha * c21;
        C[2 * ldc + 2] = alpha * c22;
        C[2 * ldc + 3] = alpha * c23;
        
        C[3 * ldc] += alpha * c30;
        C[1 + 3 * ldc] = alpha * c31;
        C[2 + 3 * ldc] = alpha * c32;
        C[3 + 3 * ldc] = alpha * c33;
}

void AddDot4x4_reg(int K, float alpha, float beta,
                  const float *A, int lda,
                  const float *B, int ldb,
                  float *C, int ldc) {
    
    float c00 = 0.0f, c01 = 0.0f, c02 = 0.0f, c03 = 0.0f,
            c10 = 0.0f, c11 = 0.0f, c12 = 0.0f, c13 = 0.0f,
            c20 = 0.0f, c21 = 0.0f, c22 = 0.0f, c23 = 0.0f,
            c30 = 0.0f, c31 = 0.0f, c32 = 0.0f, c33 = 0.0f;
    const float *a0kp, *a1kp, *a2kp, *a3kp, 
            *bk0p, *bk1p, *bk2p, *bk3p;
    bk0p = &B[0];
    bk1p = &B[ldb];
    bk2p = &B[2 * ldb];
    bk3p = &B[3 * ldb];
            
    a0kp = &A[0];
    a1kp = &A[lda];
    a2kp = &A[2 * lda];
    a3kp = &A[3 * lda];

    for (int k = 0; k < K; k++) {
        c00 += *a0kp * *bk0p;
        c01 += *a0kp * *bk1p;
        c02 += *a0kp * *bk2p;
        c03 += *a0kp++ * *bk3p;

        c10 += *a1kp * *bk0p;
        c11 += *a1kp * *bk1p;
        c12 += *a1kp * *bk2p;
        c13 += *a1kp++ * *bk3p;

        c20 += *a2kp * *bk0p;
        c21 += *a2kp * *bk1p;
        c22 += *a2kp * *bk2p;
        c23 += *a2kp++ * *bk3p;

        c30 += *a3kp * *bk0p++;
        c31 += *a3kp * *bk1p++;
        c32 += *a3kp * *bk2p++;
        c33 += *a3kp++ * *bk3p++;
    }
    
    // if (beta != 0.0f) {
    //     // 如果 beta 不为0，需要加上原来的值
    //     C[0] = alpha * c00 + beta * C[0];
    //     C[1] = alpha * c01 + beta * C[1];
    //     C[2] = alpha * c02 + beta * C[2];
    //     C[3] = alpha * c03 + beta * C[3];
        
    //     C[ldc] = alpha * c10 + beta * C[ldc];
    //     C[ldc + 1] = alpha * c11 + beta * C[ldc + 1];
    //     C[ldc + 2] = alpha * c12 + beta * C[ldc + 2];
    //     C[ldc + 3] = alpha * c13 + beta * C[ldc + 3];
        
    //     C[2 * ldc] = alpha * c20 + beta * C[2 * ldc];
    //     C[2 * ldc + 1] = alpha * c21 + beta * C[2 * ldc + 1];
    //     C[2 * ldc + 2] = alpha * c22 + beta * C[2 * ldc + 2];
    //     C[2 * ldc + 3] = alpha * c23 + beta * C[2 * ldc + 3];
        
    //     C[3 * ldc] = alpha * c30 + beta * C[3 * ldc];
    //     C[1 + 3 * ldc] = alpha * c31 + beta * C[1 + 3 * ldc];
    //     C[2 + 3 * ldc] = alpha * c32 + beta * C[2 + 3 * ldc];
    //     C[3 + 3 * ldc] = alpha * c33 + beta * C[3 + 3 * ldc];
    // } else {
        // beta 为0，直接赋值
        C[0] = alpha * c00;
        C[1] = alpha * c01;
        C[2] = alpha * c02;
        C[3] = alpha * c03;
        
        C[ldc] = alpha * c10;
        C[ldc + 1] = alpha * c11;
        C[ldc + 2] = alpha * c12;
        C[ldc + 3] = alpha * c13;
        
        C[2 * ldc] = alpha * c20;
        C[2 * ldc + 1] = alpha * c21;
        C[2 * ldc + 2] = alpha * c22;
        C[2 * ldc + 3] = alpha * c23;
        
        C[3 * ldc] = alpha * c30;
        C[1 + 3 * ldc] = alpha * c31;
        C[2 + 3 * ldc] = alpha * c32;
        C[3 + 3 * ldc] = alpha * c33;
    // }
}

/*
==========v4==========
4x4 block kernel
======================
*/
void Sgemm_v4(int M, int N, int K, 
              float alpha, 
              const float *A, int lda,
              const float *B, int ldb,
              float beta,
              float *C, int ldc) {
    
    float *BT = (float*)aligned_alloc(32, K * N * sizeof(float));
    transpose(B,BT,K,N);
    for (int i = 0; i < M; i+=4) {
        for (int j = 0; j < N; j+=4) {
            AddDot4x4_reg(K, alpha, beta,
                        &A[i * lda], lda, 
                        &BT[j * K], K, 
                        &C[i * ldc + j], ldc);
        }
    }
    free(BT);
}

#define Mb 256
#define Nb 256
#define Kb 256
#define min(a, b) ((a) < (b) ? (a) : (b))

void packA(float *packed, const float *A, int lda, int m, int k, int m_offset, int k_offset) {
    // for (int i = 0; i < m; i++) {
    //     for (int j = 0; j < k; j++) {
    //         packed[i * k + j] = A[(m_offset + i) * lda + (k_offset + j)];
    //     }
    // }
    const int SIMD_WIDTH = 16;
    
    for (int i = 0; i < m; i++) {
        const float *src_row = A + (m_offset + i) * lda + k_offset;
        float *dst_row = packed + i * k;
        
        int j = 0;
        
        for (; j + SIMD_WIDTH <= k; j += SIMD_WIDTH) {
            __m512 vec = _mm512_load_ps(src_row + j);
            _mm512_store_ps(dst_row + j, vec);
        }
        
        for (; j < k; j++) {
            dst_row[j] = src_row[j];
        }
    }
}

void packB(float *packed, const float *B, int ldb, int k, int n, int k_offset, int n_offset) {
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            packed[i * n + j] = B[(k_offset + i) * ldb + (n_offset + j)];
        }
    }
}

/*
==========v5==========
pack block
======================
*/
void Sgemm_v5(int M, int N, int K, 
              float alpha, 
              const float *A, int lda,
              const float *B, int ldb,
              float beta,
              float *C, int ldc) {
    
    float *packedA = (float*)aligned_alloc(32, Mb * Kb * sizeof(float));
    float *packedB = (float*)aligned_alloc(32, Kb * Nb * sizeof(float));

    
    for (int i = 0; i < M; i += Mb) {
        int mb = min(Mb, M - i);
        
        for (int k = 0; k < K; k += Kb) {
            int kb = min(Kb, K - k);
            
            packA(packedA, A, lda, mb, kb, i, k);
            
            for (int j = 0; j < N; j += Nb) {
                int nb = min(Nb, N - j);
                
                packB(packedB, B, ldb, kb, nb, k, j);
                
                for (int ii = 0; ii < mb; ii++) {
                    for (int jj = 0; jj < nb; jj++) {
                        float sum = 0.0f;
                        for (int kk = 0; kk < kb; kk++) {
                            sum += packedA[ii * kb + kk] * packedB[kk * nb + jj];
                        }
                        C[(i + ii) * ldc + (j + jj)] += alpha * sum;
                    }
                }
            }
        }
    }
    
    free(packedA);
    free(packedB);
}

void packBT(float *packed, const float *B, int ldb, int k, int n, int k_offset, int n_offset) {
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            packed[j * k + i] = B[(k_offset + i) * ldb + (n_offset + j)];
        }
    }
    // for (int i = 0; i < k; i++) {
    //     const int* src_ptr = &B[(k_offset + i) * ldb + n_offset];
        
    //     __m512i row = _mm512_loadu_si512((__m512i*)src_ptr);

    //     __m512i v_offsets = _mm512_set_epi32(
    //         15*k, 14*k, 13*k, 12*k, 11*k, 10*k, 9*k, 8*k,
    //         7*k, 6*k, 5*k, 4*k, 3*k, 2*k, 1*k, 0*k
    //     );
        
    //     _mm512_i32scatter_epi32(&packed[i], v_offsets, row, 4); 
    // }
    
}

void AddDot4x4_reg_pack(int K, float alpha, float beta,
                  const float *A, int lda,
                  const float *B, int ldb,
                  float *C, int ldc) {
    
    float c00 = 0.0f, c01 = 0.0f, c02 = 0.0f, c03 = 0.0f,
            c10 = 0.0f, c11 = 0.0f, c12 = 0.0f, c13 = 0.0f,
            c20 = 0.0f, c21 = 0.0f, c22 = 0.0f, c23 = 0.0f,
            c30 = 0.0f, c31 = 0.0f, c32 = 0.0f, c33 = 0.0f;
    const float *a0kp, *a1kp, *a2kp, *a3kp, 
            *bk0p, *bk1p, *bk2p, *bk3p;
    bk0p = &B[0];
    bk1p = &B[ldb];
    bk2p = &B[2 * ldb];
    bk3p = &B[3 * ldb];
            
    a0kp = &A[0];
    a1kp = &A[lda];
    a2kp = &A[2 * lda];
    a3kp = &A[3 * lda];

    for (int k = 0; k < K; k++) {
        c00 += *a0kp * *bk0p;
        c01 += *a0kp * *bk1p;
        c02 += *a0kp * *bk2p;
        c03 += *a0kp++ * *bk3p;

        c10 += *a1kp * *bk0p;
        c11 += *a1kp * *bk1p;
        c12 += *a1kp * *bk2p;
        c13 += *a1kp++ * *bk3p;

        c20 += *a2kp * *bk0p;
        c21 += *a2kp * *bk1p;
        c22 += *a2kp * *bk2p;
        c23 += *a2kp++ * *bk3p;

        c30 += *a3kp * *bk0p++;
        c31 += *a3kp * *bk1p++;
        c32 += *a3kp * *bk2p++;
        c33 += *a3kp++ * *bk3p++;
    }
    
    C[0] += alpha * c00;
    C[1] += alpha * c01;
    C[2] += alpha * c02;
    C[3] += alpha * c03;
        
    C[ldc] += alpha * c10;
    C[ldc + 1] += alpha * c11;
    C[ldc + 2] += alpha * c12;
    C[ldc + 3] += alpha * c13;
        
    C[2 * ldc] += alpha * c20;
    C[2 * ldc + 1] += alpha * c21;
    C[2 * ldc + 2] += alpha * c22;
    C[2 * ldc + 3] += alpha * c23;
        
    C[3 * ldc] += alpha * c30;
    C[1 + 3 * ldc] += alpha * c31;
    C[2 + 3 * ldc] += alpha * c32;
    C[3 + 3 * ldc] += alpha * c33;
}

/*
==========v6==========
packedBT
======================
*/
void Sgemm_v6(int M, int N, int K, 
              float alpha, 
              const float *A, int lda,
              const float *B, int ldb,
              float beta,
              float *C, int ldc) {
    
    float *packedA = (float*)aligned_alloc(32, Mb * Kb * sizeof(float));
    float *packedBT = (float*)aligned_alloc(32, Kb * Nb * sizeof(float));
    
    for (int i = 0; i < M; i += Mb) {
        int mb = min(Mb, M - i);
        
        for (int k = 0; k < K; k += Kb) {
            int kb = min(Kb, K - k);
            
            packA(packedA, A, lda, mb, kb, i, k);
            
            for (int j = 0; j < N; j += Nb) {
                int nb = min(Nb, N - j);
                
                packBT(packedBT, B, ldb, kb, nb, k, j);
                
                // for (int ii = 0; ii < mb; ii++) {
                //     for (int jj = 0; jj < nb; jj++) {
                //         float sum = 0.0f;
                //         for (int kk = 0; kk < kb; kk++) {
                //             sum += packedA[ii * kb + kk] * packedBT[kk + jj * kb];
                //         }
                //         C[(i + ii) * ldc + (j + jj)] += alpha * sum;
                //     }
                // }
                for (int ii = 0; ii < mb; ii+=4) {
                    for (int jj = 0; jj < nb; jj+=4) {
                        AddDot4x4_reg_pack(kb, alpha, beta,
                                    &packedA[ii * kb], kb, 
                                    &packedBT[jj * kb], kb, 
                                    &C[(i + ii) * ldc + (j + jj)], ldc);
                    }
                }

            }
        }
    }
    
    free(packedA);
    free(packedBT);
}

/*
==========v7==========
double buffer
======================
*/
void Sgemm_v7(int M, int N, int K,
              float alpha,
              const float *A, int lda,
              const float *B, int ldb,
              float beta,
              float *C, int ldc) {
    
    float *packedA[2];
    float *packedBT[2];
    
    packedA[0] = (float*)aligned_alloc(32, Mb * Kb * sizeof(float));
    packedA[1] = (float*)aligned_alloc(32, Mb * Kb * sizeof(float));
    packedBT[0] = (float*)aligned_alloc(32, Kb * Nb * sizeof(float));
    packedBT[1] = (float*)aligned_alloc(32, Kb * Nb * sizeof(float));
    
    if (!packedA[0] || !packedA[1] || !packedBT[0] || !packedBT[1]) {
        return;
    }
    
    for (int i = 0; i < M; i += Mb) {
        int mb = min(Mb, M - i);
        
        for (int j = 0; j < N; j += Nb) {
            int nb = min(Nb, N - j);
            
            int cur_buf = 0;
            int k_first = 1;
            packA(packedA[cur_buf], A, lda, mb, Kb, i, 0);
            packBT(packedBT[cur_buf], B, ldb, Kb, nb, 0, j);
            for (int k = 0; k < K; k += Kb) {
                int kb = min(Kb, K - k);
                int next_buf = 1 - cur_buf;
                
                if (k + Kb < K) {
                    packA(packedA[next_buf], A, lda, mb, kb, i, k + Kb);
                    packBT(packedBT[next_buf], B, ldb, kb, nb, k + Kb, j);
                }
                
                float cur_beta = (k_first && beta != 0) ? beta : 0.0f;
                for (int ii = 0; ii < mb; ii += 4) {
                    for (int jj = 0; jj < nb; jj += 4) {
                        AddDot4x4_reg_pack(kb, alpha, cur_beta,
                                    &packedA[cur_buf][ii * kb], kb,
                                    &packedBT[cur_buf][jj * kb], kb,
                                    &C[(i + ii) * ldc + (j + jj)], ldc);
                    }
                }
                
                k_first = 0;
                cur_buf = next_buf;
            }
        }
    }
    
    free(packedA[0]);
    free(packedA[1]);
    free(packedBT[0]);
    free(packedBT[1]);
}

void AddDot4x4_pack_simd(int K, float alpha, float beta,
                  const float *A, int lda,
                  const float *B, int ldb,
                  float *C, int ldc) {
    
    __m512 sum00_vec = _mm512_setzero_ps();
    __m512 sum01_vec = _mm512_setzero_ps();
    __m512 sum02_vec = _mm512_setzero_ps();
    __m512 sum03_vec = _mm512_setzero_ps();

    __m512 sum10_vec = _mm512_setzero_ps();
    __m512 sum11_vec = _mm512_setzero_ps();
    __m512 sum12_vec = _mm512_setzero_ps();
    __m512 sum13_vec = _mm512_setzero_ps();

    __m512 sum20_vec = _mm512_setzero_ps();
    __m512 sum21_vec = _mm512_setzero_ps();
    __m512 sum22_vec = _mm512_setzero_ps();
    __m512 sum23_vec = _mm512_setzero_ps();

    __m512 sum30_vec = _mm512_setzero_ps();
    __m512 sum31_vec = _mm512_setzero_ps();
    __m512 sum32_vec = _mm512_setzero_ps();
    __m512 sum33_vec = _mm512_setzero_ps();


    int k = 0;
    for (; k + 15 < K; k += 16) {
        __m512 a00_vec = _mm512_load_ps(&A[k]);
        __m512 a10_vec = _mm512_load_ps(&A[k + lda]);
        __m512 a20_vec = _mm512_load_ps(&A[k + 2 * lda]);
        __m512 a30_vec = _mm512_load_ps(&A[k + 3 * lda]);

        // 流式加载后转换为 __m512
        __m512 b00_vec = _mm512_castsi512_ps(_mm512_stream_load_si512((__m512i*)&B[k]));
        __m512 b01_vec = _mm512_castsi512_ps(_mm512_stream_load_si512((__m512i*)&B[k + ldb]));
        __m512 b02_vec = _mm512_castsi512_ps(_mm512_stream_load_si512((__m512i*)&B[k + 2*ldb]));
        __m512 b03_vec = _mm512_castsi512_ps(_mm512_stream_load_si512((__m512i*)&B[k + 3*ldb]));
        
        // 预取下一次迭代的数据到L1缓存
        _mm_prefetch(&A[k + 16], _MM_HINT_T0);
        _mm_prefetch(&A[k + 16 + lda], _MM_HINT_T0);
        _mm_prefetch(&A[k + 16 + 2 * lda], _MM_HINT_T0);
        _mm_prefetch(&A[k + 16 + 3 * lda], _MM_HINT_T0);
        
        // FMA运算
        sum00_vec = _mm512_fmadd_ps(a00_vec, b00_vec, sum00_vec);
        sum01_vec = _mm512_fmadd_ps(a00_vec, b01_vec, sum01_vec);
        sum02_vec = _mm512_fmadd_ps(a00_vec, b02_vec, sum02_vec);
        sum03_vec = _mm512_fmadd_ps(a00_vec, b03_vec, sum03_vec);

        sum10_vec = _mm512_fmadd_ps(a10_vec, b00_vec, sum10_vec);
        sum11_vec = _mm512_fmadd_ps(a10_vec, b01_vec, sum11_vec);
        sum12_vec = _mm512_fmadd_ps(a10_vec, b02_vec, sum12_vec);
        sum13_vec = _mm512_fmadd_ps(a10_vec, b03_vec, sum13_vec);

        sum20_vec = _mm512_fmadd_ps(a20_vec, b00_vec, sum20_vec);
        sum21_vec = _mm512_fmadd_ps(a20_vec, b01_vec, sum21_vec);
        sum22_vec = _mm512_fmadd_ps(a20_vec, b02_vec, sum22_vec);
        sum23_vec = _mm512_fmadd_ps(a20_vec, b03_vec, sum23_vec);

        sum30_vec = _mm512_fmadd_ps(a30_vec, b00_vec, sum30_vec);
        sum31_vec = _mm512_fmadd_ps(a30_vec, b01_vec, sum31_vec);
        sum32_vec = _mm512_fmadd_ps(a30_vec, b02_vec, sum32_vec);
        sum33_vec = _mm512_fmadd_ps(a30_vec, b03_vec, sum33_vec);
    }
    
    float c00 = _mm512_reduce_add_ps(sum00_vec);
    float c01 = _mm512_reduce_add_ps(sum01_vec);
    float c02 = _mm512_reduce_add_ps(sum02_vec);
    float c03 = _mm512_reduce_add_ps(sum03_vec);

    float c10 = _mm512_reduce_add_ps(sum10_vec);
    float c11 = _mm512_reduce_add_ps(sum11_vec);
    float c12 = _mm512_reduce_add_ps(sum12_vec);
    float c13 = _mm512_reduce_add_ps(sum13_vec);

    float c20 = _mm512_reduce_add_ps(sum20_vec);
    float c21 = _mm512_reduce_add_ps(sum21_vec);
    float c22 = _mm512_reduce_add_ps(sum22_vec);
    float c23 = _mm512_reduce_add_ps(sum23_vec);

    float c30 = _mm512_reduce_add_ps(sum30_vec);
    float c31 = _mm512_reduce_add_ps(sum31_vec);
    float c32 = _mm512_reduce_add_ps(sum32_vec);
    float c33 = _mm512_reduce_add_ps(sum33_vec);
    
    C[0] += alpha * c00;
    C[1] += alpha * c01;
    C[2] += alpha * c02;
    C[3] += alpha * c03;

    C[ldc] += alpha * c10;
    C[1 + ldc] += alpha * c11;
    C[2 + ldc] += alpha * c12;
    C[3 + ldc] += alpha * c13;

    C[2 * ldc] += alpha * c20;
    C[1 + 2 * ldc] += alpha * c21;
    C[2 + 2 * ldc] += alpha * c22;
    C[3 + 2 * ldc] += alpha * c23;

    C[3 * ldc] += alpha * c30;
    C[1 + 3 * ldc] += alpha * c31;
    C[2 + 3 * ldc] += alpha * c32;
    C[3 + 3 * ldc] += alpha * c33;
}

/*
==========v8==========
simd
======================
*/
void Sgemm_v8(int M, int N, int K,
              float alpha,
              const float *A, int lda,
              const float *B, int ldb,
              float beta,
              float *C, int ldc) {
    float *packedA[2];
    float *packedBT[2];
    
    packedA[0] = (float*)aligned_alloc(64, Mb * Kb * sizeof(float));
    packedA[1] = (float*)aligned_alloc(64, Mb * Kb * sizeof(float));
    packedBT[0] = (float*)aligned_alloc(64, Kb * Nb * sizeof(float));
    packedBT[1] = (float*)aligned_alloc(64, Kb * Nb * sizeof(float));
    
    if (!packedA[0] || !packedA[1] || !packedBT[0] || !packedBT[1]) {
        return;
    }

    int M_b=Mb;
    int N_b=Nb;
    int K_b=Kb;

    if (Mb>M) M_b = M;
    if (Nb>N) N_b = N;
    if (Kb>K) K_b = K;
    for (int i = 0; i < M; i += M_b) {
        int mb = min(M_b, M - i);
        
        for (int j = 0; j < N; j += N_b) {
            int nb = min(N_b, N - j);
            
            int cur_buf = 0;
            int k_first = 1;
            packA(packedA[cur_buf], A, lda, mb, K_b, i, 0);
            packBT(packedBT[cur_buf], B, ldb, K_b, nb, 0, j);
            for (int k = 0; k < K; k += K_b) {
                int kb = min(K_b, K - k);
                int next_buf = 1 - cur_buf;
                
                if (k + K_b < K) {
                    packA(packedA[next_buf], A, lda, mb, kb, i, k + K_b);
                    packBT(packedBT[next_buf], B, ldb, kb, nb, k + K_b, j);
                }
                
                float cur_beta = (k_first && beta != 0) ? beta : 0.0f;
                for (int ii = 0; ii < mb; ii += 4) {
                    for (int jj = 0; jj < nb; jj += 4) {
                        AddDot4x4_pack_simd(kb, alpha, cur_beta,
                                    &packedA[cur_buf][ii * kb], kb,
                                    &packedBT[cur_buf][jj * kb], kb,
                                    &C[(i + ii) * ldc + (j + jj)], ldc);
                    }
                }
                
                k_first = 0;
                cur_buf = next_buf;
            }
        }
    }
    
    free(packedA[0]);
    free(packedA[1]);
    free(packedBT[0]);
    free(packedBT[1]);
}
/*
==========v9==========
notranspose 
======================
*/

#ifndef min
#define min(a,b) (((a)<(b))?(a):(b))
#endif

#define MB_BLOCK 256
#define NB_BLOCK 256
#define KB_BLOCK 256

void packA_avx512(float *dst, const float *src, int lda, int mb, int kb) {
    __m512i vindex = _mm512_set_epi32(
        15*lda, 14*lda, 13*lda, 12*lda, 11*lda, 10*lda, 9*lda, 8*lda,
        7*lda, 6*lda, 5*lda, 4*lda, 3*lda, 2*lda, 1*lda, 0*lda
    );

    for (int i = 0; i < mb; i += 16) {
        int rows = min(16, mb - i);
        __mmask16 mask = (1ULL << rows) - 1;

        for (int k = 0; k < kb; k++) {
            __m512 res = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask, vindex, &src[i * lda + k], 4);
            _mm512_store_ps(dst, res);
            dst += 16;
        }
    }
}

void packB_avx512(float *dst, const float *src, int ldb, int kb, int nb) {
    for (int j = 0; j < nb; j += 14) {
        int cols = min(14, nb - j);
        __mmask16 mask = (1ULL << cols) - 1;

        for (int k = 0; k < kb; k++) {
            __m512 v_row = _mm512_maskz_loadu_ps(mask, &src[k * ldb + j]);
            _mm512_mask_storeu_ps(dst, mask, v_row);
            dst += 14;
        }
    }
}

void inner_kernel_16x14(int k, float alpha, float beta,
                               const float *a, const float *b,
                               float *c, int ldc, int m_rem, int n_rem) {
    __m512 c_reg[14];
    for (int i = 0; i < 14; i++) c_reg[i] = _mm512_setzero_ps();

    for (int p = 0; p < k; p++) {
        __m512 va = _mm512_load_ps(a + p * 16);
        const float *pb = b + p * 14;
        
        #pragma unroll(14)
        for (int j = 0; j < 14; j++) {
            c_reg[j] = _mm512_fmadd_ps(va, _mm512_set1_ps(pb[j]), c_reg[j]);
        }
    }

    __m512 v_alpha = _mm512_set1_ps(alpha);
    __m512 v_beta  = _mm512_set1_ps(beta);

    for (int j = 0; j < n_rem; j++) {
        float *c_ptr = &c[j];
        for (int i = 0; i < m_rem; i++) {
            float res = ((float*)&c_reg[j])[i] * alpha;
            if (beta != 0.0f) {
                c_ptr[i * ldc] = res + beta * c_ptr[i * ldc];
            } else {
                c_ptr[i * ldc] = res;
            }
        }
    }
}


void Sgemm_latest(int M, int N, int K,
                  float alpha,
                  const float *A, int lda,
                  const float *B, int ldb,
                  float beta,
                  float *C, int ldc) {
    
    size_t sizeA = ((MB_BLOCK + 15) / 16) * 16 * KB_BLOCK * sizeof(float);
    size_t sizeB = ((NB_BLOCK + 13) / 14) * 14 * KB_BLOCK * sizeof(float);

    float *pA[2], *pB[2];
    for(int i=0; i<2; i++) {
        pA[i] = (float*)aligned_alloc(64, sizeA);
        pB[i] = (float*)aligned_alloc(64, sizeB);
        if (!pA[i] || !pB[i]) return; 
    }

    for (int j = 0; j < N; j += NB_BLOCK) {
        int nb = min(NB_BLOCK, N - j);
        for (int i = 0; i < M; i += MB_BLOCK) {
            int mb = min(MB_BLOCK, M - i);
            int cur = 0;

            int kb_first = min(KB_BLOCK, K);
            packA_avx512(pA[cur], &A[i * lda], lda, mb, kb_first);
            packB_avx512(pB[cur], &B[j], ldb, kb_first, nb);
            for (int k = 0; k < K; k += KB_BLOCK) {
                int kb = min(KB_BLOCK, K - k);
                
                if (k + KB_BLOCK < K) {
                    int next_kb = min(KB_BLOCK, K - (k + KB_BLOCK));
                    packA_avx512(pA[1-cur], &A[i * lda + (k + KB_BLOCK)], lda, mb, next_kb);
                    packB_avx512(pB[1-cur], &B[(k + KB_BLOCK) * ldb + j], ldb, next_kb, nb);
                }
                

                float actual_beta = (k == 0) ? beta : 1.0f; 

                for (int jj = 0; jj < nb; jj += 14) {
                    int n_rem = min(14, nb - jj);
                    float* b_ptr = &pB[cur][(jj / 14) * kb * 14];

                    for (int ii = 0; ii < mb; ii += 16) {
                        int m_rem = min(16, mb - ii);
                        float* a_ptr = &pA[cur][(ii / 16) * kb * 16];

                        inner_kernel_16x14(kb, alpha, actual_beta,
                                           a_ptr, b_ptr,
                                           &C[(i + ii) * ldc + (j + jj)], ldc,
                                           m_rem, n_rem);
                    }
                }
                cur = 1 - cur;
            }
        }
    }

    for(int i=0; i<2; i++) { free(pA[i]); free(pB[i]); }
}