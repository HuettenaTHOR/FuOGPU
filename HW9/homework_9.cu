#include <cuda.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>

/*
For this Ben Olschar, 108021211678 and Frederik Hüttemann, 108021215247 cooperated with eachother. 
We did implement the code on our own, but agreed to using one version. The discussion and results are made in cooperation.
*/

#define CPU_NUM_ITERATIONS 10
#define GPU_NUM_ITERATIONS 1000
#define VECTOR_SIZE_MB 40
#define BLOCK_SIZE 256
#define SECTION_SIZE BLOCK_SIZE  // Must match the BLOCK_SIZE for the scan kernel


// Check the Two Results (based on lecture)
bool checkResult(float* hostRef, float* gpuRef, int length){
    
    // Allowed rounding Error
    double epsilon = 1.0E-3;
    bool match = 1;
    for (int i = 0; i < length; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon){
            match = 0;
            printf("Reductions do not match at index %d!\n", i);
            printf("host %5.2f gpu %5.2f \n", hostRef[i], gpuRef[i]);
            break;
        }
    }
    if (match) printf("Reductions match. \n");
    return match;
}


// different from lecture
// because we otherwise would run into precision errors for large summations, we fill the array with floats from 0 to 9
void initialData(float *ip, int size){
    
    time_t t;
    srand((unsigned) time(&t));
    for (int i = 0; i < size; i++) {
        // ip[i] = (float) (rand() & 0xFF) / 10.0f;
        ip[i] = (float)(i % 10); // for improved precision
    }
}

// from the lecture
double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
}


// Task 1: CPU Scan Algorithm (Baseline)
void cpuScan(float* output, float* input, int length){
    output[0] = input[0]; 
    for (int i = 1; i < length; i++){
        output[i] = output[i - 1] + input[i];
    }
}



// work efficient scan kernel from lecture with additional block_sums for arbitrary sizes
__global__ void gpu_scan_kernel(float* output, float* input, float* block_sums, int length) {
    __shared__ float XY[SECTION_SIZE];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load input into shared memory
    if (i < length) {
        XY[threadIdx.x] = input[i];
    } else {
        XY[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // Up-sweep (reduction) phase - Brent-Kung
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            XY[index] += XY[index - stride];
        }
    }
    
    // Down-sweep (distribution) phase - Brent-Kung
    for (int stride = SECTION_SIZE / 4; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < BLOCK_SIZE) {
            XY[index + stride] += XY[index];
        }
    }
    
    __syncthreads();
    
    // Write output
    if (i < length) {
        output[i] = XY[threadIdx.x];
    }
    
    // Last thread in block saves the block sum
    if (block_sums != NULL && threadIdx.x == blockDim.x - 1) {
        block_sums[blockIdx.x] = XY[threadIdx.x];
    }
}

__global__ void offset_add_kernel(float* input, float* blockSums, int N){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < N && blockIdx.x > 0){
        input[tid] += blockSums[blockIdx.x - 1];
    }
}

/*
Task 2: GPU Work-Efficient Scan for Arbitrary Sizes of n
As the lecture code only works for sizes of n that are <= BLOCK_SIZE, we need to implement a recursive approach.
We first perform a scan on each block and store the block sums in a separate array.
Then we perform a scan on the block sums array recursively.
Finally, we add the scanned block sums to each element in the corresponding blocks.

The idea is taken from a blog entry by Lukas Bierling from 10th Jan 2026:
https://medium.com/@lukasbierling/recursive-parallel-prefix-scan-using-cuda-b8181b8527a9

*/
void scan_algorithm_arbitrary_size(float* output, float* input, int length) {

    if (length <= BLOCK_SIZE) {
        gpu_scan_kernel<<<1, BLOCK_SIZE>>>(output, input, NULL, length);
        return;
    }

    int block_size = BLOCK_SIZE;
    int grid_size = (length + block_size - 1) / block_size;

    float* d_block_sums;
    float* d_block_prefix;
    cudaMalloc((void**)&d_block_sums, grid_size * sizeof(float));
    cudaMalloc((void**)&d_block_prefix, grid_size * sizeof(float));
    
    // First scan to get block sums
    gpu_scan_kernel<<<grid_size, block_size>>>(output, input, d_block_sums, length);
    cudaDeviceSynchronize();

    // Recursive scan on block sums
    scan_algorithm_arbitrary_size(d_block_prefix, d_block_sums, grid_size);
    cudaDeviceSynchronize();

    // Offset addition
    offset_add_kernel<<<grid_size, block_size>>>(output, d_block_prefix, length);
    cudaDeviceSynchronize();

    cudaFree(d_block_sums);
    cudaFree(d_block_prefix);
}

// code to run the tests
void run_test(int task) {
    switch(task) {
            
        case 1: {            
            printf("Running Task 1: CPU Scan\n");

            for (int num_elements = 100000; num_elements <= 1000000; num_elements += 100000) {
                
                size_t num_bytes = num_elements * sizeof(float);
                float *h_vector_in = (float*) malloc(num_bytes);
                float *h_vector_out = (float*) malloc(num_bytes);                
                initialData(h_vector_in, num_elements);          

                double cpu_start = cpuSecond();
                for(int i = 0; i < CPU_NUM_ITERATIONS; i++){
                    cpuScan(h_vector_out, h_vector_in, num_elements);
                }            
                double cpu_end = cpuSecond();
                double cpu_time_ms = (cpu_end - cpu_start) * 1000.0;
                double cpu_avg = cpu_time_ms / CPU_NUM_ITERATIONS;

                printf("CPU-Avg-Time for %d elements: %.8f ms\n", num_elements, cpu_avg);
            
                free(h_vector_in); 
                free(h_vector_out);
            }
            break;
        }
        
        case 2: {
            printf("Running Task 2: GPU Work-Efficient Scan\n");           
            
            for (int num_elements = 100000; num_elements <= 1000000; num_elements += 100000) {
                
                size_t num_bytes = num_elements * sizeof(float);
                float *h_vector_in = (float*) malloc(num_bytes);
                float *h_vector_out = (float*) malloc(num_bytes);           
                float *cpu_result = (float*) malloc(num_bytes);     
                initialData(h_vector_in, num_elements);          

                float *d_vector_in, *d_vector_out;
                cudaMalloc((void**)&d_vector_in, num_bytes);
                cudaMalloc((void**)&d_vector_out, num_bytes);
                cudaMemcpy(d_vector_in, h_vector_in, num_bytes, cudaMemcpyHostToDevice);

                cpuScan(cpu_result, h_vector_in, num_elements);

                double gpu_start = cpuSecond();
                for(int i = 0; i < GPU_NUM_ITERATIONS; i++){
                    scan_algorithm_arbitrary_size(d_vector_out, d_vector_in, num_elements);
                }            
                cudaDeviceSynchronize();
                double gpu_end = cpuSecond();
                double gpu_time_ms = (gpu_end - gpu_start) * 1000.0;
                double gpu_avg = gpu_time_ms / GPU_NUM_ITERATIONS;
                cudaMemcpy(h_vector_out, d_vector_out, num_bytes, cudaMemcpyDeviceToHost);
                checkResult(cpu_result, h_vector_out, num_elements);
                printf("GPU-Avg-Time for %d elements: %.8f ms\n", num_elements, gpu_avg);
                cudaFree(d_vector_in);
                cudaFree(d_vector_out);
                free(h_vector_in);
                free(h_vector_out);
                free(cpu_result);
            }
            break;
        }
        
        default:
            printf("Invalid task number.\n");
    }
}


int main() {
    run_test(1);  // CPU scan
    run_test(2);  // GPU work-efficient scan for abitrary sizes of n
    return 0;
}