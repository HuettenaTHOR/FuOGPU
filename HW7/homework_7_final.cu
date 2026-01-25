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


// Check the Two Matrices (based on lecture)
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



// TASK 1
void cpuReduction(float* vector_in, int elements, float* reduction_out){

    // Perform upscaling of the image 
    for (int i = 0; i<elements; i++){
        *reduction_out += vector_in[i];
    }
}


// TASK 2
__global__ void gpuReductionGlobal(float* vector_in, int elements, float* reduction_out){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index < elements){
        atomicAdd(reduction_out, vector_in[index]);
    }        
}


// TASK 3
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
            free(h_vector_in); free(h_vector_out);
            break;
        
        }
        case 2: {

            printf("Running Task 2: GPU reduction (global memory)\n");
            
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

            cudaMalloc((void**)&d_vector_in, bytes_in);
            cudaMalloc((void**)&d_vector_out, sizeof(float));
            
            cudaMemcpy(d_vector_in, h_vector_in, bytes_in, cudaMemcpyHostToDevice);

            // define configuration
            dim3 dimBlock(32, 1);
            dim3 dimGrid(ceil(num_elements/(float)32), 1);

            double gpu_start = cpuSecond();

            // Kernel invocation code -> run multiple times to amortize overhead
            for(int i = 0; i < GPU_NUM_ITERATIONS; i++){
                cudaMemcpy(d_vector_out, h_vector_out_gpu, sizeof(float), cudaMemcpyHostToDevice); // reset gpu_result before each iteration
                gpuReductionGlobal <<< dimGrid, dimBlock>>> (d_vector_in, num_elements, d_vector_out);
                cudaDeviceSynchronize();
            }
            
            double gpu_end = cpuSecond();
            double gpu_time_sec = gpu_end - gpu_start; // seconds
            
            double gpu_time_ms = gpu_time_sec * 1000.0; // milliseconds
            
            // Average per Iteration
            double gpu_avg = (double) (gpu_time_ms/GPU_NUM_ITERATIONS);

            cudaMemcpy(h_vector_out_gpu, d_vector_out, sizeof(float), cudaMemcpyDeviceToHost);


            bool result = checkResult(*h_vector_out, *h_vector_out_gpu);

            if (result) printf("GPU-Avg-Time global memory: %.8fms\n", gpu_avg);
            else printf("Error");
            
            cudaFree(d_vector_in); cudaFree(d_vector_out);
            free(h_vector_in), free(h_vector_out), free(h_vector_out_gpu);
            break;
        
        }
        case 3: {
            printf("Running Task 3: GPU cascaded\n");
            
            printf("Before starting, we need to get some information about the GPU launch configuration:\n");
            
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

            // use the BLOCK_SIZE and GRID_SIZE from configuration
            dim3 dimBlock(BLOCK_SIZE, 1);
            dim3 dimGrid(GRID_SIZE, 1);

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

            if (result) printf("GPU-Avg-Time global memory: %.8fms\n", gpu_avg);
            else printf("Error");
            
            cudaFree(d_vector_in); cudaFree(d_vector_out);
            free(h_vector_in), free(h_vector_out), free(h_vector_out_gpu); 
            
            break;
        }

    }
}

int main() {
    run_test(1);
    run_test(2);
    run_test(3);
    return 0;
}
