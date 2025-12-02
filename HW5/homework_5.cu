#include <cuda.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
"""
For this Ben Olschar, 108021211678 and Frederik Hüttemann, 108021215247 cooperated with eachother. 
We did implement the code on our own, but agreed to using one version. The discussion and results are made in cooperation.
"""

// define some constants
#define CPU_NUM_ITERATIONS 100
#define GPU_NUM_ITERATIONS 1000
#define WIDTH 10000
#define HEIGHT 5000
#define TILE_DIM 32
#define BLOCK_ROWS 16

// from the lecture
void initialData(float *ip, int size) {
    time_t t;
    srand((unsigned) time(&t));
    for (int i = 0; i < size; i++) {
        ip[i] = (float) (rand() & 0xFF) / 10.0f;
    }
}

// from the lecture
bool checkResult(float *hostRef, float *gpuRef, const int N) {
    double eps = 1.0E-1;
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

// task 1: CPU matrix transpose 
void CPU_transpose(float *out, float *in, const int width, const int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            out[j * height + i] = in[i * width + j];
        }
    }
}

// task 2: GPU row to row copy 
__global__ void GPU_row_to_row_copy(float *out, float *in, const int width, const int height, int nreps) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    for (int rep = 0; rep < nreps; rep++) {
        if (idx < width && idy < height) {
            out[idy * width + idx] = in[idy * width + idx];
        }
    }
}

// task 3: GPU naive matrix transpose
__global__ void GPU_naive_transpose(float *out, float *in, const int width, const int height, int nreps) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    for (int rep = 0; rep < nreps; rep++) {
        if (idx < width && idy < height) {
            out[idx * height + idy] = in[idy * width + idx];
        }
    }
}

// task 4: GPU shared memory matrix transpose but with bank conflicts as no padding is applied
__global__ void GPU_transposeCoalesced_conflicts(float *out, float *in, const int width, const int height, int nreps) {
    
    __shared__ float tile[TILE_DIM][TILE_DIM]; // no padding
    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    int index_in = xIndex + yIndex*width;
    for (int rep = 0; rep < nreps; rep++) {
        for (int i=0; i < TILE_DIM; i+=BLOCK_ROWS) {
            tile[threadIdx.y+i][threadIdx.x] = in[index_in+i*width];
        }
        __syncthreads();

        xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
        yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
        int index_out = xIndex + yIndex * height;

        for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
            if (xIndex < height && yIndex < width) {
                out[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
            }
        }
    }
}

// task 4: GPU shared memory matrix transpose with padding to avoid bank conflicts
__global__ void GPU_transposeCoalesced_padding(float *out, float *in, const int width, const int height, int nreps) {
    
    // pad the shared memory to avoid bank conflicts
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    
    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    int index_in = xIndex + yIndex*width;
    for (int rep = 0; rep < nreps; rep++) {
        for (int i=0; i < TILE_DIM; i+=BLOCK_ROWS) {
            tile[threadIdx.y+i][threadIdx.x] = in[index_in+i*width];
        }
        __syncthreads();

        xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
        yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
        int index_out = xIndex + yIndex * height;

        for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
            if (xIndex < height && yIndex < width) {
                out[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
            }
        }
    }
}

// code to run the tests
void run_test(int task) {
    switch(task) {
        case 1: {
            printf("Running Task 1: CPU matrix transpose\n");
            int width = WIDTH;
            int height = HEIGHT;
            size_t num_bytes = width * height * sizeof(float);
            float* cpu_in = (float*) malloc(num_bytes);
            float* cpu_out = (float*) malloc(num_bytes);            

            initialData(cpu_in, width*height);

            double cpu_start = cpuSecond();

            for (int i = 0; i < CPU_NUM_ITERATIONS; i++) {
                CPU_transpose(cpu_out, cpu_in, width, height);
            }
            double cpu_end = cpuSecond();
            double avg_time = (cpu_end - cpu_start) / CPU_NUM_ITERATIONS;
            printf("CPU transpose: avg. time: %f ms, bandwidth: %f GB per second\n", avg_time*1e3, (num_bytes / avg_time) / 1.0e9);            
            break;
        }
        case 2: {
            printf("Running Task 2: GPU row to row copy\n");
            int width = WIDTH;
            int height = HEIGHT;
            size_t num_bytes = width * height * sizeof(float);
            float* cpu_in = (float*) malloc(num_bytes);
            float* cpu_out = (float*) malloc(num_bytes);
            initialData(cpu_in, width*height);
            float *gpu_in, *gpu_out;
            cudaMalloc((float**)&gpu_in, num_bytes);
            cudaMalloc((float**)&gpu_out, num_bytes);
            cudaMemcpy(gpu_in, cpu_in, num_bytes, cudaMemcpyHostToDevice);
            dim3 block(16, 16);
            dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
            GPU_row_to_row_copy<<<grid, block>>>(gpu_out, gpu_in, width, height, 1); // launch the kernel once to warm up
            cudaDeviceSynchronize();
            double gpu_start = cpuSecond();            
            GPU_row_to_row_copy<<<grid, block>>>(gpu_out, gpu_in, width, height, GPU_NUM_ITERATIONS);
            cudaDeviceSynchronize();
            double gpu_end = cpuSecond();
            double avg_time = (gpu_end - gpu_start) / GPU_NUM_ITERATIONS;
            cudaMemcpy(cpu_out, gpu_out, num_bytes, cudaMemcpyDeviceToHost);
            checkResult(cpu_in, cpu_out, width*height);
            printf("GPU row to row copy: avg. time: %f ms, bandwidth: %f GB per second\n", avg_time*1e3, (num_bytes / avg_time) / 1.0e9);
            break;
        }
        case 3: {
            printf("Running Task 3: GPU naive matrix transpose\n");
            int width = WIDTH;
            int height = HEIGHT;
            size_t num_bytes = width * height * sizeof(float);
            float* cpu_in = (float*) malloc(num_bytes);
            float* cpu_out = (float*) malloc(num_bytes);
            float* cpu_out_ref = (float*) malloc(num_bytes);
            initialData(cpu_in, width*height);
            float *gpu_in, *gpu_out;
            cudaMalloc((float**)&gpu_in, num_bytes);
            cudaMalloc((float**)&gpu_out, num_bytes);
            cudaMemcpy(gpu_in, cpu_in, num_bytes, cudaMemcpyHostToDevice);
            dim3 block(16, 16);
            dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
            GPU_naive_transpose<<<grid, block>>>(gpu_out, gpu_in, width, height, 1); // launch the kernel once to warm up
            cudaDeviceSynchronize();

            // actual test
            double gpu_start = cpuSecond();            
            GPU_naive_transpose<<<grid, block>>>(gpu_out, gpu_in, width, height, GPU_NUM_ITERATIONS);
            cudaDeviceSynchronize();            
            double gpu_end = cpuSecond();
            double avg_time = (gpu_end - gpu_start) / GPU_NUM_ITERATIONS;

            cudaMemcpy(cpu_out, gpu_out, num_bytes, cudaMemcpyDeviceToHost);
            CPU_transpose(cpu_out_ref, cpu_in, width, height);
            checkResult(cpu_out_ref, cpu_out, width*height);
            printf("GPU naive transpose: avg. time: %f ms, bandwidth: %f GB per second\n", avg_time*1e3, (num_bytes / avg_time) / 1.0e9);
            break;
        }
        case 41: {
            printf("Running Task 4: GPU shared Memory transpose (without padding) \n");
            int width = WIDTH;
            int height = HEIGHT;
            size_t num_bytes = width * height * sizeof(float);
            float* cpu_in = (float*) malloc(num_bytes);
            float* cpu_out = (float*) malloc(num_bytes);
            float* cpu_out_ref = (float*) malloc(num_bytes);
            initialData(cpu_in, width*height);
            float *gpu_in, *gpu_out;
            cudaMalloc((float**)&gpu_in, num_bytes);
            cudaMalloc((float**)&gpu_out, num_bytes);
            cudaMemcpy(gpu_in, cpu_in, num_bytes, cudaMemcpyHostToDevice);
            dim3 block(TILE_DIM, BLOCK_ROWS);
            dim3 grid((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);
            GPU_transposeCoalesced_conflicts<<<grid, block>>>(gpu_out, gpu_in, width, height, 1); // launch the kernel once to warm up
            cudaDeviceSynchronize();
            // actual test
            double gpu_start = cpuSecond();            
            GPU_transposeCoalesced_conflicts<<<grid, block>>>(gpu_out, gpu_in, width, height, GPU_NUM_ITERATIONS);
            cudaDeviceSynchronize();
            
            double gpu_end = cpuSecond();
            double avg_time = (gpu_end - gpu_start) / GPU_NUM_ITERATIONS;
            cudaMemcpy(cpu_out, gpu_out, num_bytes, cudaMemcpyDeviceToHost);
            CPU_transpose(cpu_out_ref, cpu_in, width, height);
            checkResult(cpu_out_ref, cpu_out, width*height);
            printf("GPU shared memory transpose (without padding): avg. time: %f ms, bandwidth: %f GB per second\n", avg_time*1e3, (num_bytes / avg_time) / 1.0e9);
            break;
        }

        case 42: {
            printf("Running Task 4: GPU shared Memory transpose \n");
            int width = WIDTH;
            int height = HEIGHT;
            size_t num_bytes = width * height * sizeof(float);
            float* cpu_in = (float*) malloc(num_bytes);
            float* cpu_out = (float*) malloc(num_bytes);
            float* cpu_out_ref = (float*) malloc(num_bytes);
            initialData(cpu_in, width*height);
            float *gpu_in, *gpu_out;
            cudaMalloc((float**)&gpu_in, num_bytes);
            cudaMalloc((float**)&gpu_out, num_bytes);
            cudaMemcpy(gpu_in, cpu_in, num_bytes, cudaMemcpyHostToDevice);
            dim3 block(TILE_DIM, BLOCK_ROWS);
            dim3 grid((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);
            GPU_transposeCoalesced_padding<<<grid, block>>>(gpu_out, gpu_in, width, height, 1); // launch the kernel once to warm up
            cudaDeviceSynchronize();
            // actual test
            double gpu_start = cpuSecond();            
            GPU_transposeCoalesced_padding<<<grid, block>>>(gpu_out, gpu_in, width, height, GPU_NUM_ITERATIONS);
            cudaDeviceSynchronize();            
            double gpu_end = cpuSecond();
            double avg_time = (gpu_end - gpu_start) / GPU_NUM_ITERATIONS;
            cudaMemcpy(cpu_out, gpu_out, num_bytes, cudaMemcpyDeviceToHost);
            CPU_transpose(cpu_out_ref, cpu_in, width, height);
            checkResult(cpu_out_ref, cpu_out, width*height);

            printf("GPU shared memory transpose (with padding --> no bank conflicts): avg. time: %f ms, bandwidth: %f GB per second\n", avg_time*1e3, (num_bytes / avg_time) / 1.0e9);
            break;
        }
    }
}

int main() {
    run_test(1);
    run_test(2);
    run_test(3);
    run_test(41);
    run_test(42);
    return 0;
}
