#include <cuda.h>
#include <cstdio>
#include <stdio.h>


void CpuMatrixAddition(float** A, float** B, float** C, int N) 
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}


__global__ void VecAddKernel(float* A, float* B, float* C, int n) 
{
    int i = threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void VecAddWrapper(float* A, float* B, float* C, int n) 
{
    int size = n*sizeof(float);

    float *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_B, size);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_C, size);
    
    VecAddKernel <<< ceil(n/256.0), 256>>> (d_A, d_B, d_C, n);
    
    cudaMemcpy(d_C, C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}

void initialData(float *ip, int size) {
    time_t t;
    srand((unsigned) time(&t));
    for (int i = 0; i < size; i++) {
        ip[i] = (float) (rand() & 0xFF) / 10.0f;
    }
}

void checkResult(float *hostRef, float *gpuRef, const int N) {
    double eps = 1.0E-8;
    bool match = 1;
    for (int i=0; i < N; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > eps) {
            match = 0;
            printf("Arrays do not match \n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
        }
    }
    if (match) printf("Arrays match \n\n");
}

int main() 
{
    int size = 2000;
    float* data_A = (float*) malloc(size * sizeof(float));
    float* data_B = (float*) malloc(size * sizeof(float));
    float* res_1 = (float*) malloc(size * sizeof(float));
    float* res_2 = (float*) malloc(size * sizeof(float));

    initialData(data_A, size);
    initialData(data_B, size);

    VecAddWrapper(data_A, data_B, res_1, size);
    VecAddWrapper(data_A, data_B, res_2, size);

    checkResult(res_1, res_2, size);

    free(data_A);
    free(data_B);
    free(res_1);
    free(res_2);

    

}