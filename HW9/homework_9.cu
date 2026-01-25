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
#define SECTION_SIZE 512  
#define BLOCK_SIZE 256



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


// from the lecture
void initialData(float *ip, int size){
    
    time_t t;
    srand((unsigned) time(&t));
    for (int i = 0; i < size; i++) {
        ip[i] = (float) (rand() & 0xFF) / 10.0f;
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
void cpuScan(float* output, float* intput, int length){
    output[0] = 0;  
    for (int i = 1; i < length; i++){
        output[i] = output[i - 1] + intput[i - 1];
    }
}

// Task 2: GPU Work-Efficient Scan Kernel by Brent and Kung
__global__ void work_efficient_scan_kernel(float* output, float* input, int length) {
    __shared__ float XY[SECTION_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // prescan: load data into shared memory
    if (i < length) {
        XY[threadIdx.x] = input[i];
    }
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            XY[index] += XY[index - stride];        
        }
    }
    for (int stride = SECTION_SIZE / 4; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < blockDim.x) {
            XY[index + stride] += XY[index];        
        }
    }
    __syncthreads();
    // write results to global memory

    // check for dimensions (other then from the lecture)
    if (i < length) {
        output[i] = XY[threadIdx.x];
    }
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
                initialData(h_vector_in, num_elements);          

                float *d_vector_in, *d_vector_out;
                cudaMalloc((void**)&d_vector_in, num_bytes);
                cudaMalloc((void**)&d_vector_out, num_bytes);
                cudaMemcpy(d_vector_in, h_vector_in, num_bytes, cudaMemcpyHostToDevice);

                int block_size = BLOCK_SIZE;
                int grid_size = (num_elements + block_size - 1) / block_size;
                dim3 dimBlock(block_size, 1);
                dim3 dimGrid(grid_size, 1);

                double gpu_start = cpuSecond();
                for(int i = 0; i < GPU_NUM_ITERATIONS; i++){
                    work_efficient_scan_kernel<<<dimGrid, dimBlock>>>(d_vector_out, d_vector_in, num_elements);
                    cudaDeviceSynchronize();
                }            
                double gpu_end = cpuSecond();
                double gpu_time_ms = (gpu_end - gpu_start) * 1000.0;
                double gpu_avg = gpu_time_ms / GPU_NUM_ITERATIONS;

                cudaMemcpy(h_vector_out, d_vector_out, num_bytes, cudaMemcpyDeviceToHost);

                printf("GPU-Avg-Time for %d elements: %.8f ms\n", num_elements, gpu_avg);

                cudaFree(d_vector_in); 
                cudaFree(d_vector_out);
                free(h_vector_in); 
                free(h_vector_out);
            }
            break;
        }
        
        default:
            printf("Invalid task number.\n");
    }
}


int main() {
    run_test(1);  // CPU scan
    run_test(2);  // GPU work-efficient scan
    return 0;
}