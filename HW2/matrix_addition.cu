#include <cuda.h>
#include <cstdio>
#include <stdio.h>
#include <sys/time.h>
#define N_ITER 10000

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
    This code is from the lecture. The function returns true if results match, false otherwise.
*/

bool checkResult(float *hostRef, float *gpuRef, const int N) {
    double eps = 1.0E-8;
    for (int i=0; i < N; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > eps) {
            return false;
        }
    }
    return true;
}

/*
    This function returns the current cpu time in milliseconds.
    This code is from the lecture.
*/
double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
}


/*
    This function runs matrix addition on both CPU and GPU for given matrix size and block size.
    The time is taken for both CPU and GPU computations. To get an accuracte estimation, the test is run for 10.000 iterations.
    The function returns the average time for both CPU and GPU computations.
*/
bool run_test(int matrix_width, int matrix_height, int block_width, int block_height, double* cpu_avg_time, double* gpu_avg_time, bool skip_cpu=false) 
{
    // define all pointers
    float *h_A, *h_B, *h_C;
    float *h_A_GPU, *h_B_GPU, *h_C_GPU;
    float *h_C_result_from_GPU;

    // calculate size of matrix and allocate memory
    int width = matrix_width;
    int height = matrix_height;
    int size = width * height;
    int nBytes = size * sizeof(float);
    h_A = (float*) malloc(nBytes);
    h_B = (float*) malloc(nBytes);
    h_C = (float*) malloc(nBytes);

    // allocate extra memory to store result from GPU as we will compare CPU and GPU results
    h_C_result_from_GPU = (float*) malloc(nBytes);

    // allocate device memory on GPU
    cudaMalloc((float**)&h_A_GPU, nBytes);
    cudaMalloc((float**)&h_B_GPU, nBytes);
    cudaMalloc((float**)&h_C_GPU, nBytes);

    // randomly initialize data
    initialData(h_A, size);
    initialData(h_B, size);
    
    // Copy data from host to device (not part of timing)    
    cudaMemcpy(h_A_GPU, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(h_B_GPU, h_B, nBytes, cudaMemcpyHostToDevice);
    
    // Measure CPU time using std::chrono wrapper above
    double cpu_start = cpuSecond();
    if (!skip_cpu) {
        for (int i = 0; i < N_ITER; i++) {
            CpuMatrixAddition1D(h_A, h_B, h_C, width, height);
        }
    }
    double cpu_end = cpuSecond();
    double cpu_time_sec = cpu_end - cpu_start; // seconds
    double cpu_time_ms = cpu_time_sec * 1000.0; // milliseconds
    // average per iteration in milliseconds
    *cpu_avg_time = (double)(cpu_time_ms / N_ITER);    

    // calculate Block and Grid dimensions according to input (from the lecture)
    dim3 block(block_width, block_height);
    dim3 grid(ceil(width/(float)block_width), ceil(height/(float)block_height), 1);

    double gpu_start = cpuSecond();
    for (int i = 0; i < N_ITER; i++)  {
        MatAddKernel1D<<<grid, block>>>(h_A_GPU, h_B_GPU, h_C_GPU, width, height);
        cudaDeviceSynchronize();
    }
    double gpu_end = cpuSecond();
    double gpu_time = gpu_end - gpu_start;
    *gpu_avg_time = (double) gpu_time / N_ITER;

    // Copy result from device to host
    cudaMemcpy(h_C_result_from_GPU, h_C_GPU, nBytes, cudaMemcpyDeviceToHost);


    bool result = checkResult(h_C, h_C_result_from_GPU, size);

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_result_from_GPU);
    cudaFree(h_A_GPU);
    cudaFree(h_B_GPU);
    cudaFree(h_C_GPU);   
    if (skip_cpu) {
        return true;
    }
    return result;
}

int main() 
{
    /*
        Task 2 requires the following tests:
         - Matrix sizes: 10x10, 100x100, 1000x1000, 100x10000
        here, only GPU time is relevant
    */
    printf("Results for task 2:\n");

    int matrix_sizes[4][2] = { {10, 10}, {100, 100}, {1000, 1000}, {100, 10000} };
    int block_height = 16; 
    int block_width = 16;

    for (int i = 0; i < 4; i++) {
        int matrix_width = matrix_sizes[i][0];
        int matrix_height = matrix_sizes[i][1];
        double cpu_avg_time = 0.0f;
        double gpu_avg_time = 0.0f;
        bool result = run_test(matrix_width, matrix_height, block_width, block_height, &cpu_avg_time, &gpu_avg_time);
        if (result) {
            printf("Matrix Size: %dx%d, Block Size: %dx%d => CPU Avg Time: %.6f ms, GPU Avg Time: %.6f ms\n", 
                matrix_width, matrix_height, block_width, block_height, cpu_avg_time * 1000, gpu_avg_time * 1000);
        } else {
            printf("Results do not match for Matrix Size: %dx%d\n", matrix_width, matrix_height);
        }
    }
    printf("\n\n\nResults for task 4:\n");
    /* task 4 requires to test different block sizes for matrix size 100x10000 */
    int matrix_width = 100;
    int matrix_height = 10000;
    int block_sizes[3][2] = { {16, 16}, {16, 32}, {32, 16}}; 

    for (int i = 0; i < 3; i++) {
        int block_width = block_sizes[i][0];
        int block_height = block_sizes[i][1];
        double cpu_avg_time = 0.0f;
        double gpu_avg_time = 0.0f;
        bool result = run_test(matrix_width, matrix_height, block_width, block_height, &cpu_avg_time, &gpu_avg_time, true);
        if (result) {
            printf("Matrix Size: %dx%d, Block Size: %dx%d => CPU Avg Time: %.6f ms, GPU Avg Time: %.6f ms\n", 
                matrix_width, matrix_height, block_width, block_height, cpu_avg_time * 1000, gpu_avg_time * 1000);
        } else {
            printf("Results do not match for Matrix Size: %dx%d\n", matrix_width, matrix_height);
        }
    }

    return 0;
}