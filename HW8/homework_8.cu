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

int BLOCK_SIZE = 0;
int GRID_SIZE = 0;


// Check the Two Results (based on lecture)
bool checkResult(float hostRef, float gpuRef){
    
    // Allowed rounding Error
    double epsilon = 1.0E-3;
    bool match = 1;

    if (abs(hostRef-gpuRef) > epsilon){
        match = 0;
        printf("Reductions do not match!\n");
        printf("host %5.2f gpu %5.2f \n", hostRef, gpuRef);
    }

    if (match) printf("Reductions match. \n\n");
    return match;
}


// DIFFERENT FROM LECTURE CODE 
// because we ran into round / precision issues with atomicAdd on floats we decided to fill the array with 1.0f
// so that the result is exactly known and no precision issues arise
void initialData(float *ip, int size_m, int size_n){
    
    // generate diffrent seed for random number
    time_t t;
    srand((unsigned) time(&t));

    for (int i=0; i<size_m; i++){
        for (int j=0; j<size_n; j++){
            ip[i*size_n + j] = 1.0f;
        }
    }
}

// from the lecture
double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
}


// Task 1: CPU Reduction (Baseline)
void cpuReduction(float* vector_in, int elements, float* reduction_out){
    for (int i = 0; i < elements; i++){
        *reduction_out += vector_in[i];
    }
}

// Task 2: GPU Atomic Cascaded Reduction (from HW7) (copy paste from last homework)
__global__ void gpuReductionAtomicCascaded(float* vector_in, int elements, float* reduction_out){

    // Cascaded: use local register variable for thread-level sum
    float sum_thread = 0.0f;
    __shared__ float sum_block;
    
    if (threadIdx.x == 0) sum_block = 0.0f;
    __syncthreads();

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Grid-stride loop: each thread processes multiple elements (coarsening)
    for(int i = index; i < elements; i += blockDim.x * gridDim.x){
        sum_thread += vector_in[i];
    }
    
    // Block-level reduction using atomicAdd to shared memory
    atomicAdd(&sum_block, sum_thread);
    __syncthreads();

    // Only thread 0 of each block adds to global memory
    if (threadIdx.x == 0){
        atomicAdd(reduction_out, sum_block);
    }
}


// warp reduction function for last warp unrolling
__device__ void warpReduce(volatile float* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}


// task 3: Harris Cascaded Reduction with __threadfence()
// Uses Reduction #3 (Sequential Adressing), #5 (Unroll Last Warp), #7 (Multiple Adds per Thread)
__device__ unsigned int count = 0;

__global__ void gpuReductionHarrisCascaded(float* g_idata, unsigned int N, float* result) {
    
    // Shared memory for block-level reduction
    extern __shared__ float sdata[];
    __shared__ bool isLastBlockDone;
    
    unsigned int tid = threadIdx.x;
    unsigned int blockSize = blockDim.x;
   
    // Reduction #7: Multiple Adds per Thread (optimization of #4)
   
    float mySum = 0.0f;
    for (unsigned int i = blockIdx.x * (blockSize*2) + tid; i < N; i += blockSize * gridDim.x * 2) {
        mySum += g_idata[i];
        if (i + blockSize < N) {
            mySum += g_idata[i + blockSize];
        }
    }
    sdata[tid] = mySum;
    __syncthreads();

    // Reduction #3: Sequential Addressing
    for (unsigned int s = blockSize / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }    
    
    // Reduction #5: Unroll Last Warp
    if (tid < 32) {
        warpReduce(sdata, tid);
    }
    
    // Cascaded Reduction with __threadfence()
    if (tid == 0) {       
        result[blockIdx.x] = sdata[0];
        
        __threadfence();
        
        // Thread 0 of each block signals that it is done
        unsigned int value = atomicInc(&count, gridDim.x);
        
        // Thread 0 of each block determines if its block is the last block to be done
        isLastBlockDone = (value == (gridDim.x - 1));
    }
    __syncthreads();
    
    if (isLastBlockDone) {
        // The last block sums the partial sums stored in result[0..gridDim.x-1]
        float totalSum = 0.0f;        
        for (unsigned int i = tid; i < gridDim.x; i += blockSize) {
            totalSum += result[i];
        }
        sdata[tid] = totalSum;
        __syncthreads();
        
        // Final tree reduction on partial sums
        for (unsigned int s = blockSize / 2; s > 32; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        
        if (tid < 32) {
            warpReduce(sdata, tid);
        }
        
        if (tid == 0) {
            result[0] = sdata[0];
            count = 0;
        }
    }
}

// code to run the tests
void run_test(int task) {
    switch(task) {
        case 0: {
            // GPU Configuration Information
            printf("GPU Configuration Information:\n");
            int deviceId = 0;
            cudaDeviceProp props;
            cudaGetDeviceProperties(&props, deviceId);
            
            printf("GPU Device: %s\n", props.name);
            printf("Compute Capability: %d.%d\n", props.major, props.minor);
            printf("Number of SMs: %d\n", props.multiProcessorCount);
            printf("Max threads per SM: %d\n", props.maxThreadsPerMultiProcessor);
            printf("Max blocks per SM: %d\n", props.maxBlocksPerMultiProcessor);
            printf("Thread warp size: %d\n", props.warpSize);

            // Persistent blocks: Number of blocks = #MP x #resident blocks
            printf("\nFrom this we can calculate an optimal launch configuration:\n");
            printf("To maximize occupancy, we calculate BLOCK_SIZE as follows:\n");
            BLOCK_SIZE = props.maxThreadsPerMultiProcessor / props.maxBlocksPerMultiProcessor;
            printf("BLOCK_SIZE = max_threads_per_SM / max_blocks_per_SM = %d\n", BLOCK_SIZE);
            GRID_SIZE = props.multiProcessorCount * props.maxBlocksPerMultiProcessor;
            printf("GRID_SIZE = num_SMs * max_blocks_per_SM = %d (persistent blocks)\n", GRID_SIZE);

            printf("\nVector information:\n");
            printf("Vector size: %d MB\n", VECTOR_SIZE_MB);
            printf("Number of float elements: %lu\n", 
                   (unsigned long)(VECTOR_SIZE_MB * 1024 * 1024 / sizeof(float)));
            break;
        }
        
        case 1: {            
            printf("Running Task 1: CPU Reduction\n");

            int num_elements = (VECTOR_SIZE_MB * 1024 * 1024) / sizeof(float);
            int bytes_in = num_elements * sizeof(float);
            
            float *h_vector_in = (float*) malloc(bytes_in);
            float *h_vector_out = (float*) malloc(sizeof(float));
            
            *h_vector_out = 0.0f;

            initialData(h_vector_in, num_elements, 1);

            double cpu_start = cpuSecond();

            for(int i = 0; i < CPU_NUM_ITERATIONS; i++){
                *h_vector_out = 0.0f;
                cpuReduction(h_vector_in, num_elements, h_vector_out);
            }
            
            double cpu_end = cpuSecond();
            double cpu_time_ms = (cpu_end - cpu_start) * 1000.0;
            double cpu_avg = cpu_time_ms / CPU_NUM_ITERATIONS;

            printf("CPU Result: %.2f (expected: %d)\n", *h_vector_out, num_elements);
            printf("CPU-Avg-Time: %.8fms\n\n", cpu_avg);
            
            free(h_vector_in); 
            free(h_vector_out);
            break;
        }
        
        case 2: {           
            printf("Running Task 2: GPU Atomic Cascaded Reduction (from HW7)\n");            
            int num_elements = (VECTOR_SIZE_MB * 1024 * 1024) / sizeof(float);
            int bytes_in = num_elements * sizeof(float);
            
            float *h_vector_in = (float*) malloc(bytes_in);
            float *h_vector_out = (float*) malloc(sizeof(float));
            float *h_vector_out_gpu = (float*) malloc(sizeof(float));
            
            *h_vector_out = 0.0f;
            *h_vector_out_gpu = 0.0f;
            
            initialData(h_vector_in, num_elements, 1);
            cpuReduction(h_vector_in, num_elements, h_vector_out);

            float *d_vector_in, *d_vector_out;

            cudaMalloc((void**)&d_vector_in, bytes_in);
            cudaMalloc((void**)&d_vector_out, sizeof(float));
            
            cudaMemcpy(d_vector_in, h_vector_in, bytes_in, cudaMemcpyHostToDevice);

            dim3 dimBlock(BLOCK_SIZE, 1);
            dim3 dimGrid(GRID_SIZE, 1);

            double gpu_start = cpuSecond();

            for(int i = 0; i < GPU_NUM_ITERATIONS; i++){
                cudaMemcpy(d_vector_out, h_vector_out_gpu, sizeof(float), cudaMemcpyHostToDevice);
                gpuReductionAtomicCascaded<<<dimGrid, dimBlock>>>(d_vector_in, num_elements, d_vector_out);
                cudaDeviceSynchronize();
            }
            
            double gpu_end = cpuSecond();
            double gpu_time_ms = (gpu_end - gpu_start) * 1000.0;
            double gpu_avg = gpu_time_ms / GPU_NUM_ITERATIONS;

            cudaMemcpy(h_vector_out_gpu, d_vector_out, sizeof(float), cudaMemcpyDeviceToHost);

            bool result = checkResult(*h_vector_out, *h_vector_out_gpu);

            if (result) printf("GPU-Avg-Time (Atomic Cascaded): %.8fms\n\n", gpu_avg);
            else printf("Error in GPU Atomic Cascaded Reduction\n\n");
            
            cudaFree(d_vector_in); 
            cudaFree(d_vector_out);
            free(h_vector_in); 
            free(h_vector_out); 
            free(h_vector_out_gpu);
            break;
        }
        
        case 3: {
            // task 3: Harris Cascaded with threadfence            
            printf("Running Task 3: Harris Cascaded Algorithm with threadfence\n");
            
            int num_elements = (VECTOR_SIZE_MB * 1024 * 1024) / sizeof(float);
            int bytes_in = num_elements * sizeof(float);
            
            float *h_vector_in = (float*) malloc(bytes_in);
            float *h_vector_out = (float*) malloc(sizeof(float));
            float *h_vector_out_gpu = (float*) malloc(sizeof(float));
            
            *h_vector_out = 0.0f;
            *h_vector_out_gpu = 0.0f;
            
            initialData(h_vector_in, num_elements, 1);
            cpuReduction(h_vector_in, num_elements, h_vector_out);

            float *d_vector_in, *d_result;

            cudaMalloc((void**)&d_vector_in, bytes_in);
            cudaMalloc((void**)&d_result, GRID_SIZE * sizeof(float));
            
            cudaMemcpy(d_vector_in, h_vector_in, bytes_in, cudaMemcpyHostToDevice);

            dim3 dimBlock(BLOCK_SIZE, 1);
            dim3 dimGrid(GRID_SIZE, 1);
            size_t sharedMemSize = BLOCK_SIZE * sizeof(float);

            double gpu_start = cpuSecond();

            for(int i = 0; i < GPU_NUM_ITERATIONS; i++){
                unsigned int zero = 0;
                cudaMemcpyToSymbol(count, &zero, sizeof(unsigned int));
                
                gpuReductionHarrisCascaded<<<dimGrid, dimBlock, sharedMemSize>>>(
                    d_vector_in, num_elements, d_result);
                cudaDeviceSynchronize();
            }
            
            double gpu_end = cpuSecond();
            double gpu_time_ms = (gpu_end - gpu_start) * 1000.0;
            double gpu_avg = gpu_time_ms / GPU_NUM_ITERATIONS;

            cudaMemcpy(h_vector_out_gpu, d_result, sizeof(float), cudaMemcpyDeviceToHost);

            bool result = checkResult(*h_vector_out, *h_vector_out_gpu);

            if (result) printf("GPU-Avg-Time (Harris Cascaded + threadfence): %.8fms\n\n", gpu_avg);
            else printf("Error in Harris Cascaded Reduction\n\n");
            
            cudaFree(d_vector_in); 
            cudaFree(d_result);
            free(h_vector_in); 
            free(h_vector_out); 
            free(h_vector_out_gpu);
            break;
        }
        
        default:
            printf("Invalid task number.\n");
    }
}


int main() {
    run_test(0);  // GPU Configuration
    run_test(1);  // CPU Reduction (Baseline)
    run_test(2);  // GPU Atomic Cascaded (from HW7)
    run_test(3);  // Harris Cascaded with threadfence
    return 0;
}
