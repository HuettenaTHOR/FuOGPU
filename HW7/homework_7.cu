#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
/*
For this Ben Olschar, 108021211678 and Frederik Hüttemann, 108021215247 cooperated with eachother. 
We did implement the code on our own, but agreed to using one version. The discussion and results are made in cooperation.
*/

// define some constants
#define GPU_NUM_ITERATIONS 1000
#define CPU_NUM_ITERATIONS 100

// resulting floats
static float cpu_result = 0.0f;
__device__ float gpu_result = 0.0f;
__device__ float gpu_result_cascaded = 0.0f;

int BLOCK_SIZE = 0;
int GRID_SIZE = 0;

// DIFFERENT FROM LECTURE CODE 
// because we ran into round / precision issues with atomicAdd on floats we decided to fill the array with 1.0f
// so that the result is exactly known and no precision issues arise
void initialData(float *ip, int size) {
    for (int i = 0; i < size; i++) {
        ip[i] = (float) 1.0f;
    }
}

// from the lecture
bool checkResult(float *hostRef, float *gpuRef, const int N) {
    double eps = 1.0E-5;  
    for (int i=0; i < N; i++) {
        if (fabsf(hostRef[i] - gpuRef[i]) > eps) {
            printf("Result mismatch at index %d: host %f gpu %f\n", i, hostRef[i], gpuRef[i]);
            return false;
        }
    }
    printf("Results match.\n");
    return true;
}

// from the lecture
double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
}

static void cpu_reduction(float *input, int num_elements) {
    for (int i = 0; i < num_elements; i++) {
        cpu_result += input[i];
    }
  
}

__global__ void gpu_reduction(float *input, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < num_elements; i += blockDim.x * gridDim.x) {
        atomicAdd(&gpu_result, input[i]); 
    }
}

__global__ void gpu_reduction_cascaded(float *input, int num_elements) {

    // sum reduction on thread level (aggregation + coarsening)
    float thread_sum = 0.0f; // each thread has its own sum
    __shared__ float block_sum; // shared memory for block level reduction
    block_sum = 0.0f;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < num_elements; i += blockDim.x * gridDim.x) {
        thread_sum += input[i];
    }

    // sub reduction on block level (using shared memory)    
    atomicAdd(&block_sum, thread_sum);
    __syncthreads();

    // final reduction to global memory (only one thread per block)
    if (threadIdx.x == 0) {
        atomicAdd(&gpu_result_cascaded, block_sum);
    }
}

void run_tests(int task) {
    switch(task) {
        case 0: {
            printf("Before starting, we need to get some information about the GPU (RTX 4060ti: 8.9 Compute Capability) launch configuration:\n");
            
            int deviceId = 0;
            cudaDeviceProp props;
            cudaGetDeviceProperties(&props, deviceId);
            printf("Number of SMs: %d\n", props.multiProcessorCount);
            printf("Max threads per SM: %d\n", props.maxThreadsPerMultiProcessor);
            printf("Max blocks per SM: %d\n", props.maxBlocksPerMultiProcessor);
            printf("Thread warp size: %d\n", props.warpSize);

            printf("From this we can calculate an optimal launch configuration:\n");
            printf("To maximize occupancy, we calculate BLOCK_SIZE as follows:\n");
            BLOCK_SIZE = props.maxThreadsPerMultiProcessor / props.maxBlocksPerMultiProcessor;
            printf("Suggested BLOCK_SIZE: %d\n", BLOCK_SIZE);
            GRID_SIZE = props.multiProcessorCount * props.maxBlocksPerMultiProcessor;
            printf("Suggested GRID_SIZE: %d\n", GRID_SIZE);

            printf("We also need to know the number of elements in our 40 MB float array: \n");
            printf("A 40 MB array contains %lu float elements.\n", 40 * 1024 * 1024 / sizeof(float));
            printf("\n\n");

            break;
        }
        case 1: {
            printf("Running Task 1: CPU Reduction:\n");
            int float_size = 40; // 40 MB
            int num_float = float_size * 1024 * 1024 / sizeof(float);
            size_t bytes = num_float * sizeof(float);
            float *input = (float*) malloc(bytes);

            initialData(input, num_float);
            // no loop for timing because the reset of cpu_result is pretty complicated for GPU
            double start = cpuSecond();                         
            for (int i = 0; i < CPU_NUM_ITERATIONS; i++) {    
                cpu_result = 0.0f; // reset result before each iteration
                cpu_reduction(input, num_float);
            }            
            double end = cpuSecond();
            printf("CPU Reduction Time: %f ms\n", ((end - start) / CPU_NUM_ITERATIONS) * 1000.0f);
            free(input);
            break;
        }
        case 2: {
            printf("Running Task 2: GPU Reduction Test with atomic add on global memory:\n");
            int float_size = 40; // 40 MB
            int num_float = float_size * 1024 * 1024 / sizeof(float);            
            size_t bytes = num_float * sizeof(float);
            float *cpu_input = (float*) malloc(bytes);
            float *gpu_input;           

            // fill array with 1.0f so we know the result and dont run into precision issues           
            initialData(cpu_input, num_float);             
            
            cudaMalloc((void**)&gpu_input, bytes);

            cudaMemcpy(gpu_input, cpu_input, bytes, cudaMemcpyHostToDevice);
            cpu_result = 0.0f; // reset result before CPU reduction
            cpu_reduction(cpu_input, num_float);

            // because we use a different approach here, we need to adjust the launch configuration
            int threads_per_block = 64;
            int num_blocks = num_float / (threads_per_block * 16); // 16 elements per thread
            dim3 block(threads_per_block, 1);
            dim3 grid(num_blocks, 1);

            float reset_value = 0.0f; // reset value for gpu_result

            // single GPU run as the reseting of gpu_result is complicated
            double start = cpuSecond();   

            for (int i = 0; i < GPU_NUM_ITERATIONS; i++) {
                cudaMemcpyToSymbol(gpu_result, &reset_value, sizeof(float)); // reset gpu_result before each iteration
                gpu_reduction<<<grid, block>>>(gpu_input, num_float);
                cudaDeviceSynchronize();
            }
            double end = cpuSecond();

            float *gpu_compare_result = (float*) malloc(sizeof(float));
            cudaMemcpyFromSymbol(gpu_compare_result, gpu_result, sizeof(float));
            checkResult(&cpu_result, gpu_compare_result, 1);
            printf("GPU Reduction Time: %f ms\n", ((end - start) / GPU_NUM_ITERATIONS) * 1000.0f);
            cudaFree(gpu_input);
            free(cpu_input);          
            break;
        }
        case 3: {
            printf("Running Task 3: GPU Reduction Test with cascaded reduction:\n");
            

            int float_size = 40; // 40 MB
            int num_float = float_size * 1024 * 1024 / sizeof(float);            
            size_t bytes = num_float * sizeof(float);

            float *cpu_input = (float*) malloc(bytes);
            float *gpu_input;           

            initialData(cpu_input, num_float); // fill array with 1.0f so we know the result and dont run into precision issues           
            
            cudaMalloc((void**)&gpu_input, bytes);

            cudaMemcpy(gpu_input, cpu_input, bytes, cudaMemcpyHostToDevice);
            cpu_result = 0.0f; // reset result before CPU reduction
            cpu_reduction(cpu_input, num_float);

            float reset_value = 0.0f; // reset value for gpu_result_cascaded

            // use the BLOCK_SIZE and GRID_SIZE from configuration task
            dim3 block(BLOCK_SIZE, 1);
            dim3 grid(GRID_SIZE, 1);

            double start = cpuSecond();
            for (int i = 0; i < GPU_NUM_ITERATIONS; i++) {
                cudaMemcpyToSymbol(gpu_result_cascaded, &reset_value, sizeof(float)); // reset gpu_result before each iteration
                gpu_reduction_cascaded<<<grid, block>>>(gpu_input, num_float);
                cudaDeviceSynchronize();
            }
            double end = cpuSecond();
            printf("GPU Cascaded Reduction Time: %f ms\n", ((end - start) / GPU_NUM_ITERATIONS) * 1000.0f);

            float *gpu_compare_result = (float*) malloc(sizeof(float));
            cudaMemcpyFromSymbol(gpu_compare_result, gpu_result_cascaded, sizeof(float));
            checkResult(&cpu_result, gpu_compare_result, 1);       
            cudaFree(gpu_input);
            free(cpu_input);          
            break;
        }        
        default:
            printf("Invalid task number.\n");
    }
}

int main() {
    run_tests(0);
    run_tests(1);
    run_tests(2);
    run_tests(3);
    return 0;
}