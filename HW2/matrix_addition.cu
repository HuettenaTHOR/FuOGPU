#include <cuda.h>
#include <cstdio>
#include <stdio.h>

/*
    CPU implementation of matrix addition
    Here we have a 2D matrix represented as a 1D array as we can map all 2D matrices into a 
    1D array by using index flattening.
*/
void CpuMatrixAddition1D(float* A, float* B, float* C, int width, int height) 
{
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int idx = i * width + j;
            C[idx] = A[idx] + B[idx];
        }
    }
}

/*
    GPU implementation of vector addition in 1D array representation. Similar to the 
    CPU implementation, we assume the 2D matrix flattened into a 1D array 
    and perform addition.
    From the lecture, we get:
     - ix = threadIdx.x + blockIdx.x * blockDim.x
     - iy = threadIdx.y + blockIdx.y * blockDim.y
    So the index in the flattened array is:
        - idx = iy * width + ix
    
    On each computation, we need to check if the index is within bounds.
    The bounds are:
     - ix < width (lecture it is col < n)
     - iy < height (lecture it is row < m)
*/
__global__ void MatAddKernel1D(float* A, float* B, float* C, int width, int height) 
{
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int idx = iy * width + ix;
    if (iy < height && ix < width) {
        C[idx] = A[idx] + B[idx];
    }
}

/*
    Function to initialize data for testing.
    This code is from the lecture.
*/

void initialData(float *ip, int size) {
    time_t t;
    srand((unsigned) time(&t));
    for (int i = 0; i < size; i++) {
        ip[i] = (float) (rand() & 0xFF) / 10.0f;
    }
}

/*
    Function to check the result of the GPU computation.
    This code is from the lecture.
*/

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
    float *h_A, *h_B, *h_C;
    float *h_A_GPU, *h_B_GPU, *h_C_GPU;
    float *h_C_result_from_GPU;

    int width = 200;
    int height = 200;
    int size = width * height;
    int nBytes = size * sizeof(float);
    h_A = (float*) malloc(nBytes);
    h_B = (float*) malloc(nBytes);
    h_C = (float*) malloc(nBytes);
    h_C_result_from_GPU = (float*) malloc(nBytes);

    cudaMalloc((float**)&h_A_GPU, nBytes);
    cudaMalloc((float**)&h_B_GPU, nBytes);
    cudaMalloc((float**)&h_C_GPU, nBytes);
    printf("Matrix Addition of size %d x %d\n", width, height);
    initialData(h_A, size);
    initialData(h_B, size);
    printf("Data initialization done.\n");
    /*
    Copy data from host to device
    */
    cudaMemcpy(h_A_GPU, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(h_B_GPU, h_B, nBytes, cudaMemcpyHostToDevice);
    printf("Data copy to device done.\n");

    /*
    Run CPU matrix addition
    */
    CpuMatrixAddition1D(h_A, h_B, h_C, width, height);
    printf("CPU matrix addition done.\n");

    /*
    Run GPU matrix addition
    */

    int block_width = 16;
    int block_height = 16;
    dim3 block(block_width, block_height);
    dim3 grid(ceil(width/(float)block_width), ceil(height/(float)block_height), 1);
    MatAddKernel1D<<<grid, block>>>(h_A_GPU, h_B_GPU, h_C_GPU, width, height);
    printf("GPU matrix addition done.\n");
    cudaDeviceSynchronize();

    /*
    Copy result from device to host
    */
    cudaMemcpy(h_C_result_from_GPU, h_C_GPU, nBytes, cudaMemcpyDeviceToHost);
    printf("Data copy to host done.\n");

    /*
    Check results
    */
    checkResult(h_C, h_C_result_from_GPU, size);
    printf("Result check done.\n");
    /*
    Free memory
    */
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(h_A_GPU);
    cudaFree(h_B_GPU);
    cudaFree(h_C_GPU);

    return 0;
}