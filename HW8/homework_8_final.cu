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

// Check the two Results (based on lecture)
bool checkResult(float hostRef, float gpuRef){
    
    // Allowed rounding Error
    double epsilon = 1.0E-3;
    bool match = 1;

    if(isnan(gpuRef)){
        match = 0;
        printf("Reductions do not match!\n");
        printf("host %5.2f gpu %5.2f \n", hostRef, gpuRef);
    }

    if (abs(hostRef-gpuRef) > epsilon){
        match = 0;
        printf("Reductions do not match!\n");
        printf("host %5.2f gpu %5.2f \n", hostRef, gpuRef);
    }
    
    printf("host %5.2f gpu %5.2f \n", hostRef, gpuRef);
    if (match) printf("Reductions match.\n");
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



// TASK 1
void cpuReduction(float* vector_in, int elements, float* reduction_out){

    // Perform upscaling of the image 
    for (int i = 0; i<elements; i++){
        *reduction_out += vector_in[i];
    }
}




// Copy and Paste for Comparisson
__global__ void gpuReductionCascaded(float* vector_in, int elements, float* reduction_out){

    // sum reduction on thread level (aggregation + coarsening)
    float sum_thread = 0.0; // each thread has its own sum
    __shared__ float sum_block; // shared memory for block level reduction
    sum_block = 0.0f;
    __syncthreads();

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index < elements){
        for(int i = index; i < elements; i += blockDim.x * gridDim.x){
            sum_thread += vector_in[i];
        }
        __syncthreads();
        
        // sub reduction on block level (using shared memory
        atomicAdd(&sum_block, sum_thread);
        __syncthreads();

        // final reduction to global memory (only one thread per block)
        if (threadIdx.x == 0){
            atomicAdd(reduction_out, sum_block);
        }
    }
    
    
}


// Task 2
//Reduction #5: Unroll last warp
template <unsigned int blockSize>
__device__ void warpReduce(volatile float* sdata, unsigned int tid){

    if(blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if(blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if(blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if(blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if(blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if(blockSize >= 2) sdata[tid] += sdata[tid + 1];

}

__device__ unsigned int count = 0;
template <unsigned int blockSize>
__global__ void gpuReduceHarris(float* g_idata, float* g_odata, unsigned int n){

    __shared__ bool isLastBlockDone;
    extern __shared__ float sdata[];

    //Reduction #7: Multiple Adds / Threads
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize*2) + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0.0f;

    while (i < n){
        sdata[tid] += g_idata[i] + g_idata[i+blockSize];
        i += gridSize;
    }
    __syncthreads();

    //Reduction #6: Completely Unroll for-loop
    if (blockSize >= 512){
        if(tid<256){sdata[tid] += sdata[tid + 256];}
        __syncthreads();
    }
    if (blockSize >= 256){
        if(tid<128){sdata[tid] += sdata[tid + 128];}
        __syncthreads();
    }
    if (blockSize >= 128){
        if(tid<64){sdata[tid] += sdata[tid + 64];}
        __syncthreads();
    }
    if(tid < 32) warpReduce<blockSize>(sdata, tid);
    
    // usage of thread-fence and atomic operation to determine last block
    if (tid == 0){
        g_odata[blockIdx.x] = sdata[0];

        //using thread-fence
        __threadfence();

        unsigned int value = atomicInc(&count, gridDim.x);
        isLastBlockDone = (value == (gridDim.x - 1));

    } 

    __syncthreads();

    // Final reduction in Kernel using all the optimizations above
    if(isLastBlockDone){
        float totalSum = 0.0f;
        unsigned int sum_i = tid;

        // each thread sums multiple values from g_odata
        while (sum_i < gridDim.x){
            totalSum += g_odata[sum_i];
            sum_i += blockSize;
        }

        sdata[tid] = totalSum; 
        __syncthreads();

        if (blockSize >= 512){
            if(tid<256){sdata[tid] += sdata[tid + 256];}
            __syncthreads();
        }
        if (blockSize >= 256){
            if(tid<128){sdata[tid] += sdata[tid + 128];}
            __syncthreads();
        }
        if (blockSize >= 128){
            if(tid<64){sdata[tid] += sdata[tid + 64];}
            __syncthreads();
        }
        if(tid < 32) warpReduce<blockSize>(sdata, tid);

        // final write to global memory
        if (threadIdx.x == 0){
            g_odata[0]= sdata[0];
            count = 0;
        }
    }

}

// code to run the tests
void run_test(int task) {
    switch(task) {      

        case 1: {
            printf("Running Task 1: CPU Reduction\n");

            int num_elements = (VECTOR_SIZE_MB * 1024 * 1024) / sizeof(float);
            int bytes_in =num_elements * sizeof(float);
            
            float *h_vector_in = (float*) malloc(bytes_in);
            float *h_vector_out = (float*) malloc(sizeof(float));
            
            *h_vector_out = 0.0f;

            initialData(h_vector_in, num_elements, 1);

            // Timer for CPU
            double cpu_start = cpuSecond();

            // CPU-Reduction -> run multiple times to amortize overhead
            for(int i = 0;  i< CPU_NUM_ITERATIONS; i++){
                *h_vector_out = 0.0f; // reset result before each iteration
                cpuReduction(h_vector_in, num_elements, h_vector_out);
            }
            
            double cpu_end = cpuSecond();
            double cpu_time_sec = cpu_end - cpu_start; // seconds
            
            double cpu_time_ms = cpu_time_sec * 1000.0; // milliseconds
            
            // Average per Iteration
            double cpu_avg = (double) (cpu_time_ms/CPU_NUM_ITERATIONS);

            printf("CPU-Avg-Time : %.8fms\n", cpu_avg);
            printf("\n");
            free(h_vector_in); free(h_vector_out);
            break;
        
        }
        case 2: {

            printf("Running Task 2: GPU cascaded\n");

            int num_elements = (VECTOR_SIZE_MB * 1024 * 1024) / sizeof(float);
            int bytes_in =num_elements * sizeof(float);
            
            float *h_vector_in = (float*) malloc(bytes_in);
            float *h_vector_out = (float*) malloc(sizeof(float));
            float *h_vector_out_gpu = (float*) malloc(sizeof(float));
            
            *h_vector_out = 0.0f;
            *h_vector_out_gpu = 0.0f;

             // fill array with 1.0f so we know the result and dont run into precision issues 
            initialData(h_vector_in, num_elements, 1);
            
            cpuReduction(h_vector_in, num_elements, h_vector_out);
            
            float *d_vector_in, *d_vector_out;

            cudaMalloc((void**)&d_vector_in, bytes_in);
            cudaMalloc((void**)&d_vector_out, sizeof(float));
            
            cudaMemcpy(d_vector_in, h_vector_in, bytes_in, cudaMemcpyHostToDevice);

            // use the BLOCK_SIZE and GRID_SIZE from configuration of the last homework
            dim3 dimBlock(64, 1);
            dim3 dimGrid(640, 1);

            double gpu_start = cpuSecond();

            // Kernel invocation code -> run multiple times to amortize overhead
            for(int i = 0; i < GPU_NUM_ITERATIONS; i++){
                cudaMemcpy(d_vector_out, h_vector_out_gpu, sizeof(float), cudaMemcpyHostToDevice); // reset gpu_result before each iteration
                gpuReductionCascaded <<< dimGrid, dimBlock>>> (d_vector_in, num_elements, d_vector_out);
                cudaDeviceSynchronize();
            }
            
            double gpu_end = cpuSecond();
            double gpu_time_sec = gpu_end - gpu_start; // seconds
            
            double gpu_time_ms = gpu_time_sec * 1000.0; // milliseconds
            
            // Average per Iteration
            double gpu_avg = (double) (gpu_time_ms/GPU_NUM_ITERATIONS);

            cudaMemcpy(h_vector_out_gpu, d_vector_out, sizeof(float), cudaMemcpyDeviceToHost);


            bool result = checkResult(*h_vector_out, *h_vector_out_gpu);
            cudaDeviceSynchronize();

            if (result) printf("GPU-Avg-Time atomic Reduction: %.8fms\n", gpu_avg);
            else printf("Error");
            printf("\n");
            cudaFree(d_vector_in); cudaFree(d_vector_out);
            free(h_vector_in), free(h_vector_out), free(h_vector_out_gpu); 
            
            break;
        }
        case 3: {

            printf("Running Task 3: GPU Harris\n");
            int deviceId = 0;
            cudaDeviceProp props;
            cudaGetDeviceProperties(&props, deviceId);
            printf("Getting Device Properties:\n");
            printf("Number of SMs: %d\n", props.multiProcessorCount);
            printf("Max Threads per SM: %d\n", props.maxThreadsPerMultiProcessor);
            
            int num_elements = (VECTOR_SIZE_MB * 1024 * 1024) / sizeof(float);
            int bytes_in = num_elements * sizeof(float);
            
            float *h_vector_in = (float*) malloc(bytes_in);
            float *h_vector_out = (float*) malloc(sizeof(float));
            float *h_vector_out_gpu = (float*) malloc(sizeof(float));
            
            
            *h_vector_out = 0.0f;
            *h_vector_out_gpu = 0.0f;

             // fill array with 1.0f so we know the result and dont run into precision issues 
            initialData(h_vector_in, num_elements, 1);
            
            cpuReduction(h_vector_in, num_elements, h_vector_out);
            
            float *d_vector_in, *d_vector_out;

            // setting the block size to 512 for maximum occupancy
            int HARRIS_BLOCK_SIZE = 512;
            // calculating grid size based on device properties (using the formula from the discussion)
            int HARRIS_GRID_SIZE = props.multiProcessorCount * (props.maxThreadsPerMultiProcessor/HARRIS_BLOCK_SIZE);
            
            int smemSize = HARRIS_BLOCK_SIZE * sizeof(float);
            dim3 dimBlock(HARRIS_BLOCK_SIZE, 1);
            dim3 dimGrid(HARRIS_GRID_SIZE, 1);

            cudaMalloc((void**)&d_vector_in, bytes_in);
            cudaMalloc((void**)&d_vector_out, HARRIS_BLOCK_SIZE * sizeof(float));
            float *h_zeros = (float*) calloc(HARRIS_GRID_SIZE, sizeof(float));
            
            cudaMemcpy(d_vector_in, h_vector_in, bytes_in, cudaMemcpyHostToDevice);

            
            double gpu_start = cpuSecond();

            // Kernel invocation code -> run multiple times to amortize overhead
            for(int i = 0; i < GPU_NUM_ITERATIONS; i++){
                cudaMemcpy(d_vector_out, h_zeros, HARRIS_GRID_SIZE * sizeof(float), cudaMemcpyHostToDevice); // reset gpu_result before each iteration
                
                // Switch-Case for block_sizes
                switch(HARRIS_BLOCK_SIZE) {
                    case 512:
                        gpuReduceHarris<512><<< dimGrid, dimBlock, smemSize>>> (d_vector_in, d_vector_out, num_elements);
                        break;
                    case 256:
                        gpuReduceHarris<256><<< dimGrid, dimBlock, smemSize>>> (d_vector_in, d_vector_out, num_elements);
                        break;
                    case 128:
                        gpuReduceHarris<128><<< dimGrid, dimBlock, smemSize>>> (d_vector_in, d_vector_out, num_elements);
                        break;
                    case 64:
                        gpuReduceHarris<64><<< dimGrid, dimBlock, smemSize>>> (d_vector_in, d_vector_out, num_elements);
                        break;
                    case 32:
                        gpuReduceHarris<32><<< dimGrid, dimBlock, smemSize>>> (d_vector_in, d_vector_out, num_elements);
                        break;
                    case 16:
                        gpuReduceHarris<16><<< dimGrid, dimBlock, smemSize>>> (d_vector_in, d_vector_out, num_elements);
                        break;
                    case 8:
                        gpuReduceHarris<8><<< dimGrid, dimBlock, smemSize>>> (d_vector_in, d_vector_out, num_elements);
                        break;
                    case 4:
                        gpuReduceHarris<4><<< dimGrid, dimBlock, smemSize>>> (d_vector_in, d_vector_out, num_elements);
                        break;
                    case 2:
                        gpuReduceHarris<2><<< dimGrid, dimBlock, smemSize>>> (d_vector_in, d_vector_out, num_elements);
                }
                cudaDeviceSynchronize();
            }
            
            double gpu_end = cpuSecond();
            double gpu_time_sec = gpu_end - gpu_start; // seconds
            
            double gpu_time_ms = gpu_time_sec * 1000.0; // milliseconds
            
            // Average per Iteration
            double gpu_avg = (double) (gpu_time_ms/GPU_NUM_ITERATIONS);

            cudaMemcpy(h_vector_out_gpu, d_vector_out, sizeof(float), cudaMemcpyDeviceToHost);


            bool result = checkResult(*h_vector_out, *h_vector_out_gpu);
            cudaDeviceSynchronize();

            if (result) printf("GPU-Avg-Time Harris: %.8fms\n", gpu_avg);
            else printf("Error Harris");
            printf("\n");
            cudaFree(d_vector_in); cudaFree(d_vector_out);
            free(h_vector_in); free(h_vector_out); free(h_vector_out_gpu); free(h_zeros);
            
            break;
        }
    }
}

int main() {
    run_test(0);
    run_test(1);
    run_test(2);
    run_test(3);
    return 0;
}
