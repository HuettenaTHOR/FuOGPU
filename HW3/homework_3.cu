#include <cuda.h>
#include <cstdio>
#include <stdio.h>
#include <sys/time.h>
#include <assert.h>


#define NUM_ITERATIONS 512*1024

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

void cpu_global_ilp1(float *vec, size_t ilp_num)
{
    for (int k = 0; k < ilp_num; k++)
    {
#pragma unroll 16
        for (int i = 0; i < NUM_ITERATIONS; i++)
        {
            size_t base = 1 * k;
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
            size_t base =  4 * k;
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
    vec[4 * (blockIdx.x * blockDim.x + threadIdx.x) + 0] = a0;
    vec[4 * (blockIdx.x * blockDim.x + threadIdx.x) + 1] = a1;
    vec[4 * (blockIdx.x * blockDim.x + threadIdx.x) + 2] = a2;
    vec[4 * (blockIdx.x * blockDim.x + threadIdx.x) + 3] = a3;
}

__global__ void gpu_global_ilp1(float *vec, size_t ilp_num) 
{
#pragma unroll 16
    for (int i = 0; i < NUM_ITERATIONS; i++) 
    {
        size_t base = 1 * (blockIdx.x * blockDim.x + threadIdx.x);
        vec[base + 0] *= 3;
    }
}

__global__ void gpu_global_ilp4(float *vec, size_t ilp_num) 
{
#pragma unroll 16
    for (int i = 0; i < NUM_ITERATIONS; i++) 
    {
        size_t base = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
        vec[base + 0] *= 3;
        vec[base + 1] *= 3;
        vec[base + 2] *= 3;
        vec[base + 3] *= 3;
    }
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

void run_ilp(float* result, bool gpu = true, bool global_mem = false, bool ilp4 = false)
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

    const size_t max_blocks = 2048;
    const size_t vector_size = blockDim.x * blockDim.y * max_blocks * unrolling;

    const size_t vector_size_bytes = vector_size * sizeof(float);

    float* cpu_vec_gpu = (float*)malloc(vector_size_bytes);
    float* cpu_vec = (float*)malloc(vector_size_bytes);
    float* gpu_vec;
    cudaMalloc((void**)&gpu_vec, vector_size_bytes);

    cudaMemcpy(cpu_vec_gpu, gpu_vec, vector_size_bytes, cudaMemcpyDeviceToHost);

    for (int i = 1; i <= max_blocks; i = i + 16) {
        dim3 gridDim(i, 1);
        size_t n = gridDim.x * gridDim.y * blockDim.x * blockDim.y;
    // account for unrolling (ILP) and two FLOPs per a = a*b + c (mul + add)
    size_t total_op = unrolling * NUM_ITERATIONS * n;

        gpu_global_ilp1<<<1, 1>>>(gpu_vec, 0);
        // cpu computation
        if (!gpu) {
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
            printf("cpu/%s/%s/%zu: %zu MFLOP/s\n", global_mem ? "global" : "local", ilp4 ? "ilp4" : "ilp1",(size_t) n, (size_t)flops);
            *result = *cpu_vec;
        } else {
            // gpu computation
            t_start = cpuSecond();
            if (!global_mem && !ilp4) {
                gpu_ilp1<<<gridDim, blockDim>>>(gpu_vec, a, b, c, n);
            } else if (!global_mem && ilp4) {
                gpu_ilp4<<<gridDim, blockDim>>>(gpu_vec, a, b, c, n);
            } else if (global_mem && ilp4) {
                gpu_global_ilp4<<<gridDim, blockDim>>>(gpu_vec, n);
            } else {
                gpu_global_ilp1<<<gridDim, blockDim>>>(gpu_vec, n);
            }
            cudaGetLastError();
            cudaDeviceSynchronize();
            t_end = cpuSecond();
            gpu_time = t_end - t_start;
            flops = (1e-6 * total_op) / gpu_time;
            printf("gpu/%s/%s/%zu: %zu MFLOP/s\n", global_mem ? "global" : "local", ilp4 ? "ilp4" : "ilp1", n, (size_t)flops);

            cudaMemcpy(cpu_vec_gpu, gpu_vec, vector_size_bytes, cudaMemcpyDeviceToHost);
            *result = *cpu_vec_gpu;
        }
    }
    cudaFree(gpu_vec);
    free(cpu_vec);
    free(cpu_vec_gpu);
}


int main() {
    float* result_cpu;
    float* result_gpu;

    // even if this is not the best way to do it, allocate here to avoid reallocation in each run
    result_cpu = (float*)malloc(2048 * 16 * 4 * sizeof(float));
    result_gpu = (float*)malloc(2048 * 16 * 4 * sizeof(float));

    // compare gpu_ilp1 and cpu_ilp1 local
    // gpu=False for cpu, global_mem=False for local memory, ilp4=False for ilp1
    printf("ILP1, local, CPU \n");
    run_ilp(result_cpu, false, false, false);
    
    // gpu=True for gpu, global_mem=False for local memory, ilp4=False for ilp1
    printf("ILP1, local, GPU \n");
    run_ilp(result_gpu, true, false, false);

    printf("results match: %s\n", checkResult(result_cpu, result_gpu, 2048 * 16 * 1) ? "true" : "false");

    // // gpu=false for cpu, global_mem=True for global memory, ilp4=False for ilp1
    printf("ILP1, global, CPU \n");
    run_ilp(result_cpu, false, true, false);

    // // gpu=True for gpu, global_mem=True for global memory, ilp4=False for ilp1
    printf("ILP1, global, GPU \n");
    run_ilp(result_gpu, true, true, false);

    printf("results match: %s\n", checkResult(result_cpu, result_gpu, 2048 * 16 * 1) ? "true" : "false");

    // gpu=False for cpu, global_mem=False for local memory, ilp4=True for ilp4
    printf("ILP4, local, CPU \n");
    run_ilp(result_cpu, false, false, true);

    // gpu=True for gpu, global_mem=False for local memory, ilp4=True for ilp4
    printf("ILP4, local, GPU \n");
    run_ilp(result_gpu, true, false, true);

    printf("results match: %s\n", checkResult(result_cpu, result_gpu, 2048 * 16 * 4) ? "true" : "false");

    // gpu=false for cpu, global_mem=True for global memory, ilp4=True for ilp4
    printf("ILP4, global, CPU \n");
    run_ilp(result_cpu, false, true, true);

    // gpu=True for gpu, global_mem=True for global memory, ilp4=True for ilp4
    printf("ILP4, global, GPU \n");
    run_ilp(result_gpu, true, true, true);

    printf("results match: %s\n", checkResult(result_cpu, result_gpu, 2048 * 16 * 4) ? "true" : "false");

}  