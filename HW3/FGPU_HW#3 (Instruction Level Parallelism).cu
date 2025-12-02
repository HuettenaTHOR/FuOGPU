// For this Homework I (Ben Olschar 108021211678) worked in cooperation with Frederik Hütteman (108021215247).
// We did implement the code on our own, but agreed to using one version.

#include <stdio.h>
#include <sys/time.h>

// Number of Iterations per Configuration
#define NUM_ITERATIONS 1000000
// Number of Executions per Configuration (for average time)
#define EXEC_NUM 1000
// Number of arithmetic instructions used in on kernel (Task 2)
#define NUM_ARITHMETICS 4

// Kernel calc1 and calc2 are working without using any global memory (TASK 1)
// This Kernel only does one arithmetic instruction 
__global__ void calc1(float *d_result_a){

    float a = 1.0;
    float b = 2.0;
    float c = 3.0;
    
    #pragma unroll 16
    for(int i=0; i < NUM_ITERATIONS; i++) {
        a = a * b + c;
    }
    d_result_a[0] = a;
}

// This Kernel does four arithmetic instructions 
__global__ void calc2(float *d_result_b){

    float a = 1.0;
    float b = 2.0;
    float c = 3.0;
    float d = 4.0;
    float e = 5.0;
    float f = 6.0;

    #pragma unroll 16
    for(int i=0; i < NUM_ITERATIONS; i++) {
        a = a * b + c; d = d * b + c; e = e * b + c; f = f * b + c;
    }
    d_result_b[0] = a;
    d_result_b[1] = d;
    d_result_b[2] = e;
    d_result_b[3] = f;
}



// Kernel calc3 and calc4 are working with global memory (TASK 2)
// This Kernel only does one arithmetic instruction
// See Version 1 in lecture
__global__ void calc3(float *vec){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_ITERATIONS) vec[idx] *= 3;
}

// This Kernel does four arithmetic instructions and therefor calculates a 4 times bigger vector
// See Version 2 in lecture
__global__ void calc4(float *vec, int num_arithmetics){

    for(int i=0; i < num_arithmetics; i++) {
        int idx = blockIdx.x * (num_arithmetics*blockDim.x) + num_arithmetics*threadIdx.x + i;
        if (idx < NUM_ITERATIONS * NUM_ARITHMETICS) vec[idx] *= 3;
    }   
}


// Get current CPU-Time (based lecture)
double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
}

void initialData(float *ip, int size){

    // generate diffrent seed for random number
    time_t t;
    srand((unsigned) time(&t));

    for (int i=0; i<size; i++){
        ip[i] = (float) (rand() & 0xFF ) / 10.0f; // rand() & 0xFF -> 0..255 ; (rand() & 0xFF ) / 10.0f; -> 0.1 -> 25.5
    }

}


// Calculate Kernels with different block-sizes and therefore with different amounts of threads
bool run_calculation(int block_height, int block_width, float* result_a, float* result_b, float* result_c, float* result_d, double* calc_1_avg, double* calc_2_avg, double* calc_3_avg, double* calc_4_avg){


    // Allocate needed memory 
    float *d_result_a, *d_result_b, *d_result_c, *d_result_d;;
    cudaMalloc((void**)&d_result_a, sizeof(float));
    cudaMalloc((void**)&d_result_b, 4 * sizeof(float));
    cudaMalloc((void**)&d_result_c, NUM_ITERATIONS * sizeof(float));
    cudaMalloc((void**)&d_result_d, NUM_ARITHMETICS * NUM_ITERATIONS * sizeof(float));

    // copy inital vectos from host to device (Task 2)
    cudaMemcpy(d_result_c, result_c, NUM_ITERATIONS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result_d, result_d, NUM_ARITHMETICS * NUM_ITERATIONS * sizeof(float), cudaMemcpyHostToDevice);


    // Define Block size and Grid (Task 1) -> Number of threads = block_width * block_height * 1 * 1
    dim3 dimBlock(block_width, block_height);
    dim3 dimGrid(1,1);


    // Time for first Kernel
    double calc_1_start = cpuSecond();

    // Run-Calculations Kenrel with 1 arithmetic instruction (non global memory)
    for(int i = 0;  i< EXEC_NUM; i++){
        calc1 <<< dimGrid, dimBlock>>> (d_result_a);
        cudaDeviceSynchronize();
    }
    
    
    double calc_1_end = cpuSecond();
    double calc_1_sec = calc_1_end - calc_1_start; // seconds
    double calc_1_ms = calc_1_sec * 1000.0; // milliseconds

    // Average per Iteration
    *calc_1_avg = (double) (calc_1_ms/EXEC_NUM);

    cudaMemcpy(result_a, d_result_a, sizeof(float), cudaMemcpyDeviceToHost);


    // Time for second Kernel
    double calc_2_start = cpuSecond();

    // Run-Calculations Kenrel with 4 arithmetic instruction (non global memory)
    for(int i = 0;  i< EXEC_NUM; i++){
        calc2 <<< dimGrid, dimBlock>>> (d_result_b);
        cudaDeviceSynchronize();
    }
    
    double calc_2_end = cpuSecond();
    double calc_2_sec = calc_2_end - calc_2_start; // seconds
    double calc_2_ms = calc_2_sec * 1000.0; // milliseconds

    // Average per Iteration
    *calc_2_avg = (double) (calc_2_ms/EXEC_NUM);

    cudaMemcpy(result_b, d_result_b, 4 * sizeof(float), cudaMemcpyDeviceToHost);


    // Define new Grid-Dim to meet Problem size of "NUM_ITERATIONS"
    // Define Grid size (Task 2) -> Number of threads ~ NUM_ITERATIONS
    dimGrid = dim3(ceil((float)NUM_ITERATIONS/(float)(block_width*block_height)), 1);

    // Time for third Kernel
    double calc_3_start = cpuSecond();

    // Run-Calculations Kenrel with 1 arithmetic instruction (global memory)
    for(int i = 0;  i< EXEC_NUM; i++){
        calc3 <<< dimGrid, dimBlock>>> (d_result_c);
        cudaDeviceSynchronize();
    }
    
    double calc_3_end = cpuSecond();
    double calc_3_sec = calc_3_end - calc_3_start; // seconds
    double calc_3_ms = calc_3_sec * 1000.0; // milliseconds

    // Average per Iteration
    *calc_3_avg = (double) (calc_3_ms/EXEC_NUM);

    cudaMemcpy(result_c, d_result_c, NUM_ITERATIONS * sizeof(float), cudaMemcpyDeviceToHost);


    // Time for third Kernel
    double calc_4_start = cpuSecond();

    // Run-Calculations Kenrel with 4 arithmetic instruction (global memory)
    for(int i = 0;  i< EXEC_NUM; i++){
        calc4 <<< dimGrid, dimBlock>>> (d_result_d, NUM_ARITHMETICS);
        cudaDeviceSynchronize();
    }
    
    double calc_4_end = cpuSecond();
    double calc_4_sec = calc_4_end - calc_4_start; // seconds
    double calc_4_ms = calc_4_sec * 1000.0; // milliseconds

    // Average per Iteration
    *calc_4_avg = (double) (calc_4_ms/EXEC_NUM);

    cudaMemcpy(result_d, d_result_d, NUM_ITERATIONS * sizeof(float), cudaMemcpyDeviceToHost);

    // Free Memory
    cudaFree(d_result_a); cudaFree(d_result_b); cudaFree(d_result_c); cudaFree(d_result_d);

    return true;
}


int main (void){
    
    // Allocate needed storage
    float *result_A = (float*) malloc(sizeof(float));
    float *result_B = (float*) malloc(4 * sizeof(float));

    const int bytes =  NUM_ITERATIONS * sizeof(float);

    float *result_C = (float*) malloc(bytes);
    float *result_D = (float*) malloc(NUM_ARITHMETICS * bytes);

    initialData(result_C, NUM_ITERATIONS);
    initialData(result_D, NUM_ARITHMETICS * NUM_ITERATIONS);
    
    // run calculations with diffrent block sizes and with that diffrent amounts of threads per block
    int block_sizes[11][2] = {{16, 4}, {16, 6}, {16, 8}, {16, 10}, {16, 12}, {16, 16}, {16, 24}, {16, 32}, {16, 64}, {16, 92}, {16, 128}};

    printf("\n\nResults with multiple different block-sizes: \n\n");

    for(int i = 0; i<11; i++){
        double calc_1_avg = 0.0f;
        double calc_2_avg = 0.0f;
        double calc_3_avg = 0.0f;
        double calc_4_avg = 0.0f;

        bool result = run_calculation( block_sizes[i][0], block_sizes[i][1], result_A, result_B, result_C, result_D, &calc_1_avg, &calc_2_avg, &calc_3_avg, &calc_4_avg);
        cudaDeviceSynchronize();

        if (result) printf("Calculations Done for Block-Sizes: %d x %d; calc_1-Time: %.8fms and calc_2-Time: %.8fms (non global memory)\nCalculations Done for Block-Sizes: %d x %d; calc_3-Time: %.8fms and calc_4-Time: %.8fms (global memory)\n\n", block_sizes[i][0], block_sizes[i][1], calc_1_avg, calc_2_avg, block_sizes[i][0], block_sizes[i][1], calc_3_avg, calc_4_avg);
        else printf("Error");
    }

    // Free memory
    free(result_A); free(result_B); free(result_C); free(result_D);
    return 0; 
}