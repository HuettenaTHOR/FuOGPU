#include <cuda.h>
#include <cstdio>
#include <stdio.h>
#include <sys/time.h>
#include <assert.h>


#define NUM_ITERATIONS 1024*1024
#define NUM_REPETITIONS 512

void cpu_ilp1(float *vec, float a, float b, float c, size_t ilp_num)
{
    for (int k = 0; k < ilp_num; k++) 
    {
#pragma unroll 16
        for (int i = 0; i < NUM_ITERATIONS; i++) 
        {
            a = a * b + c;
        }
        // "store in global memory"
        size_t base = 1 * k;
        vec[base + 0] = a;
    }    
}

void cpu_ilp4(float *vec, float a, float b, float c, size_t ilp_num)
{
    for(int k = 0; k < ilp_num; k++)
    {
        for (int k = 0; k < ilp_num; k++) {
            float a0 = a;
            float a1 = a;
            float a2 = a;
            float a3 = a;
        
#pragma unroll 16
            for(int i = 0; i < NUM_ITERATIONS; i++)
            {
                a0 = a0 * b + c;
                a1 = a1 * b + c;
                a2 = a2 * b + c;
                a3 = a3 * b + c;
            }
            size_t base = 4 * k;
            //store globally
            vec[base + 0] = a0;
            vec[base + 1] = a1;
            vec[base + 2] = a2;
            vec[base + 3] = a3;
        }
    }
}

void cpu_global_ilp1(float *vec, size_t ilp_num)
{
    for (int k = 0; k < ilp_num; k++)
    {
#pragma unroll 16
        for (int i = 0; i < NUM_ITERATIONS; i++)
        {
            size_t base = 1 * (i * ilp_num + k);
            vec[base + 0] *= 3;
        }
    }
}

void cpu_global_ilp4(float *vec, size_t ilp_num)
{
    for (int k = 0; k < ilp_num; k++)
    {
#pragma unroll 16
        for (int i = 0; i < NUM_ITERATIONS; i++)
        {
            size_t base =  4 * (i*ilp_num + k);
            vec[base + 0] *= 3;
            vec[base + 1] *= 3;
            vec[base + 2] *= 3;
            vec[base + 3] *= 3;
        }
    }
}


__global__ void gpu_ilp1(float *vec, float a_in, float b, float c, size_t ilp_num) 
{
    float a = a_in;
#pragma unroll 16
        for (int i = 0; i < NUM_ITERATIONS; i++) 
        {
            a = a * b + c;
        }
    
    // "store in global memory"
    vec[1 * (blockIdx.x * blockDim.x + threadIdx.x) + 0] = a;   
}
__global__ void gpu_ilp4(float *vec, float a_in, float b, float c, size_t ilp_num) 
{
    float a0 = a_in;
    float a1 = a_in;
    float a2 = a_in;
    float a3 = a_in;
#pragma unroll 16
        for (int i = 0; i < NUM_ITERATIONS; i++) 
        {
            a0 = a0 * b + c;
            a1 = a1 * b + c;
            a2 = a2 * b + c;
            a3 = a3 * b + c;
        }
    
    // "store in global memory"
    vec[1 * (blockIdx.x * blockDim.x + threadIdx.x) + 0] = a0;
    vec[1 * (blockIdx.x * blockDim.x + threadIdx.x) + 1] = a1;
    vec[1 * (blockIdx.x * blockDim.x + threadIdx.x) + 2] = a2;
    vec[1 * (blockIdx.x * blockDim.x + threadIdx.x) + 3] = a3;
}


bool checkResult(float *hostRef, float *gpuRef, const int N) {
    double eps = 1.0E-8;
    for (int i=0; i < N; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > eps) {
            return false;
        }
    }
    return true;
}

double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
}

void run_ilp(bool global_mem = false, bool ilp4 = false)
{
    double cpu_time;
    double gpu_time;
    float a = 1.0f;
    float b = 2.0f;
    float c = 3.0f;

    double t_start, t_end, flops;
    size_t unrolling;
    dim3 blockDim(16, 1);
    if (ilp4) {
        unrolling = 4;
    } else {
        unrolling = 1;
    }

    const size_t max_blocks = 256;
    const size_t vector_size = blockDim.x * blockDim.y * max_blocks * unrolling;

    const size_t vector_size_bytes = vector_size * sizeof(float);

    float* cpu_vec_gpu = (float*)malloc(vector_size_bytes);
    float* cpu_vec = (float*)malloc(vector_size_bytes);
    float* gpu_vec;
    cudaMalloc((void**)&gpu_vec, vector_size_bytes);

    gpu_ilp1<<<1, 1>>>(gpu_vec, 0, 0, 0, 0);


    cudaMemcpy(cpu_vec_gpu, gpu_vec, vector_size_bytes, cudaMemcpyDeviceToHost);

    for (int i = 1; i <= max_blocks; i = i * 2) {
        dim3 gridDim(i, 1);
        size_t n = gridDim.x * gridDim.y * blockDim.x * blockDim.y;
        size_t total_op = 1 * NUM_ITERATIONS * n;


        // cpu computation
        
        t_start = cpuSecond();
        if (!global_mem && !ilp4) {
            cpu_ilp1(cpu_vec, a, b, c, n);
        } else if (!global_mem && ilp4) {
            cpu_ilp4(cpu_vec, a, b, c, n);
        } else if (global_mem && ilp4) {
            cpu_global_ilp4(cpu_vec, n);
        } else {
            cpu_global_ilp1(cpu_vec, n);
        }
        t_end = cpuSecond();
        cpu_time = t_end - t_start;
        flops =( 1e-6 * total_op) / (cpu_time);
        printf("cpu/%s/%s: %zu MFLOP/s\n", global_mem ? "global" : "local", ilp4 ? "ilp4" : "ilp1", (size_t)flops);

        // gpu computation
        t_start = cpuSecond();
        if (!global_mem && !ilp4) {
            gpu_ilp1<<<gridDim, blockDim>>>(gpu_vec, a, b, c, n);
        } else if (!global_mem && ilp4) {
            gpu_ilp4<<<gridDim, blockDim>>>(gpu_vec, a, b, c, n);
        } else if (global_mem && ilp4) {
            // gpu_global_ilp4<<<gridDim, blockDim>>>(gpu_vec, n);
        } else {
            // gpu_global_ilp1<<<gridDim, blockDim>>>(gpu_vec, n);
        }
        cudaGetLastError();
        cudaDeviceSynchronize();
        t_end = cpuSecond();
        gpu_time = t_end - t_start;
        flops = (1e-6 * total_op) / gpu_time;
        printf("gpu/%s/%s/%zu: %zu MFLOP/s\n", global_mem ? "global" : "local", ilp4 ? "ilp4" : "ilp1", n, (size_t)flops);
    }



    if (!checkResult(cpu_vec, cpu_vec_gpu, vector_size)) {
        fprintf(stderr, "Results do not match!\n");
        exit(EXIT_FAILURE);
    }

    cudaFree(gpu_vec);
    free(cpu_vec);
    free(cpu_vec_gpu);
}


int main() {
    // compare gpu_ilp1 and cpu_ilp1 local
    // run_ilp(false, false); 
    // compare gpu_ilp4 and cpu_ilp4 local
    run_ilp(false, true);    

}  