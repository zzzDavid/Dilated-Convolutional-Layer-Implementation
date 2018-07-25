#ifndef GEMM_H
#define GEMM_H

void gemm_bin(int M, int N, int K, float ALPHA, 
        char  *A, int lda, 
        float *B, int ldb,
        float *C, int ldc);
        
void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
                    float *A, int lda, 
                    float *B, int ldb,
                    float BETA,
                    float *C, int ldc);

void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc);

#ifdef GPU
void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A_gpu, int lda, 
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc);

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc);

void group_gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA,
                    float *A_gpu, int lda, int offsetA,
                    float *B_gpu, int ldb, int offset_B,
                    float BETA,
                    float *C_gpu, int ldc, int offset_C, int group);

void ztSgmvStrideBatched(const float * __restrict__ x_gpu, int K,
		const float * __restrict__ A_gpu, int N, int groups, float * __restrict__ C_gpu);
#endif
#endif
