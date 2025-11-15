#include <cuda.h>
#include <cstdio>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>

#define TILEWIDTH 16

void initialData(float *ip, int size) {
    time_t t;
    srand((unsigned) time(&t));
    for (int i = 0; i < size; i++) {
        ip[i] = (float) (rand() & 0xFF) / 10.0f;
    }
}

bool checkResult(float *hostRef, float *gpuRef, const int N) {
    double eps = 1.0E-2;
    for (int i=0; i < N; i++) {
        if (fabsf(hostRef[i] - gpuRef[i]) > eps) {
            printf("Result mismatch at index %d: host %f gpu %f\n", i, hostRef[i], gpuRef[i]);
            return false;
        }
    }
    printf("Results match.\n");
    return true;
}

double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
}

// CPU: multiply A (A_rows x A_cols) * B (A_cols x B_cols) = C (A_rows x B_cols)
void matrixMultiplicationCPU(float* A, float* B, float* C, int A_rows, int A_cols, int B_cols) {
    for (int row = 0; row < A_rows; row++) {
        for (int col = 0; col < B_cols; col++) {
            float value = 0;
            for (int k = 0; k < A_cols; k++) {
                value += A[row * A_cols + k] * B[k * B_cols + col];
            }
            C[row * B_cols + col] = value;
        }
    }
}

// GPU: multiply d_M (M_rows x common_dim) * d_N (common_dim x P_cols) = d_P (M_rows x P_cols)
__global__ void matrixMultiplicationGPU_non_shared(float* d_M, float* d_N, float* d_P, int common_dim, int M_rows, int P_cols) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    if ((row < M_rows) && (col < P_cols)) {
        float pValue = 0;
        for (int k = 0; k < common_dim; ++k) {
            pValue += d_M[row*common_dim + k] * d_N[k*P_cols + col];
        }
        d_P[row*P_cols + col] = pValue;
    }
}

__global__ void matrixMultiplicationGPUTiled(float* d_M, float* d_N, float* d_P, int common_dim, int M_rows, int P_cols) {
    __shared__ float Mds[TILEWIDTH][TILEWIDTH];
    __shared__ float Nds[TILEWIDTH][TILEWIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILEWIDTH + ty;
    int col = bx * TILEWIDTH + tx;

    float p_value = 0;

    for (int m = 0; m < (common_dim + TILEWIDTH - 1) / TILEWIDTH; ++m) {

        int tiledCol = m * TILEWIDTH + tx;
        int tiledRow = m * TILEWIDTH + ty;
        
        if (row < M_rows && tiledCol < common_dim) {
            Mds[ty][tx] = d_M[row * common_dim + tiledCol];
        } else {
            Mds[ty][tx] = 0.0f;
        }
        
        if (tiledRow < common_dim && col < P_cols) {
            Nds[ty][tx] = d_N[tiledRow * P_cols + col];
        } else {
            Nds[ty][tx] = 0.0f;
        }
        
        __syncthreads();

        for (int k = 0; k < TILEWIDTH; ++k) {
            p_value += Mds[ty][k] * Nds[k][tx];
        }
    }        
    if (row < M_rows && col < P_cols) {
        d_P[row * P_cols + col] = p_value;
    }
}


void run_test(int m_width, int n_height, int common_dim) {

    int p_width = n_height;   // number of cols in P
    int p_height = m_width;   // number of rows in P

    size_t m_bytes = (size_t)m_width * common_dim * sizeof(float);
    size_t n_bytes = (size_t)common_dim * n_height * sizeof(float);
    size_t p_bytes = (size_t)p_width * p_height * sizeof(float);

    float* data_M = (float*) malloc(m_bytes);
    float* data_N = (float*) malloc(n_bytes);
    float* data_P = (float*) malloc(p_bytes);

    float* gpu_M;
    float* gpu_N;
    float* gpu_P;
    float* gpu_P_result = (float*) malloc(p_bytes);

    cudaMalloc((void**)&gpu_M, m_bytes);
    cudaMalloc((void**)&gpu_N, n_bytes);
    cudaMalloc((void**)&gpu_P, p_bytes);

    initialData(data_M, m_width*common_dim);
    initialData(data_N, common_dim*n_height);

    cudaMemcpy(gpu_M, data_M, m_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_N, data_N, n_bytes, cudaMemcpyHostToDevice);


    printf("running cpu computation\n");
    matrixMultiplicationCPU(data_M, data_N, data_P, m_width, common_dim, p_width); // use correct dims
    printf("finished\n");


    /*
    ===================
    this is non shared
    ===================


    int block_width = 16;
    int block_height = 16;
    dim3 block(block_width, block_height);
    // grid uses result matrix dimensions (P_cols x P_rows)
    dim3 grid((int)ceil(p_width/(float)block_width), (int)ceil(p_height/(float)block_height), 1);

    matrixMultiplicationGPU_non_shared<<<grid, block>>>(gpu_M, gpu_N, gpu_P, common_dim, m_width, p_width);
    */

    dim3 block(TILEWIDTH, TILEWIDTH);
    // grid must cover result matrix dimensions (P_cols x P_rows)
    dim3 grid((p_width + TILEWIDTH - 1) / TILEWIDTH, (p_height + TILEWIDTH - 1) / TILEWIDTH);

    matrixMultiplicationGPUTiled<<<grid, block>>>(gpu_M, gpu_N, gpu_P, common_dim, m_width, p_width);

    // copy device -> host: destination first
    cudaMemcpy(gpu_P_result, gpu_P, p_bytes, cudaMemcpyDeviceToHost);

    checkResult(data_P, gpu_P_result, p_width*p_height);

    free(data_M);
    free(data_N);
    free(data_P);
    free(gpu_P_result);

    cudaFree(gpu_M);
    cudaFree(gpu_N);
    cudaFree(gpu_P);    

}
int main() {
    run_test(10000, 20000, 5000);
}