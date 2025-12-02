// For this Ben Olschar, 108021211678 and Frederik Hüttemann, 
// 108021215247 cooperated with eachother. We did implement the code on our own, 
// but agreed to using one version. 


#include <cuda.h>
#include <cstdio>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>

// defining constants
#define TILEWIDTH 16
#define CPU_TILEWIDTH 16

// define different number of iterations for CPU and GPU, as CPU is slower
#define CPU_NUM_ITERATIONS 10
#define GPU_NUM_ITERATIONS 100

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


/*
CPU implementations of matrix multiplication (non-shared and shared memory versions)
*/

// CPU: multiply A (A_rows x A_cols) * B (A_cols x B_cols) = C (A_rows x B_cols)
void matrixMultiplicationCPU_non_shared(float* A, float* B, float* C, int A_rows, int A_cols, int B_cols) {
    for (int row = 0; row < A_rows; row++) {
        for (int col = 0; col < B_cols; col++) {
            float value = 0;
            for (int k = 0; k < A_cols; k++) {
                value += A[row * A_cols + k] * B[k * B_cols + col];
            }
            C[row * B_cols + col] = value;
        }
    }
}

/*
CPU matrix multiplication using tiling (shared memory concept)
This is expected to be slower than the non-shared version on CPU due to the overhead of copying data
*/
void matrixMultiplicationCPU_shared(float* A, float* B, float* C, int A_rows, int A_cols, int B_cols) {
    // Use small local tile buffers (CPU_TILEWIDTH is a compile-time constant)
    for (int row = 0; row < A_rows; row++) {
        for (int col = 0; col < B_cols; col++) {
            float value = 0.0f;
            int numTiles = (A_cols + CPU_TILEWIDTH - 1) / CPU_TILEWIDTH;
            for (int m = 0; m < numTiles; m++) {
                int kStart = m * CPU_TILEWIDTH;
                int kEnd = kStart + CPU_TILEWIDTH;
                if (kEnd > A_cols) kEnd = A_cols;
                int tLen = kEnd - kStart;

                // local buffers representing the "tile" for this row and column
                float Mtile_local[CPU_TILEWIDTH];
                float Ntile_local[CPU_TILEWIDTH];

                // load the tile (from A: elements of this row; from B: elements of these rows at this column)
                for (int k = 0; k < tLen; k++) {
                    Mtile_local[k] = A[row * A_cols + (kStart + k)];
                    Ntile_local[k] = B[(kStart + k) * B_cols + col];
                }

                // accumulate product over this partial/full tile
                for (int k = 0; k < tLen; k++) {
                    value += Mtile_local[k] * Ntile_local[k];
                }
            }
            C[row * B_cols + col] = value;
        }
    }
}

/*
GPU implementations of matrix multiplication (non-shared and shared memory versions)
*/
__global__ void matrixMultiplicationGPU_non_shared(float* d_M, float* d_N, float* d_P, int common_dim, int M_rows, int P_cols) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    if ((row < M_rows) && (col < P_cols)) {
        float pValue = 0;
        for (int k = 0; k < common_dim; ++k) {
            pValue += d_M[row*common_dim + k] * d_N[k*P_cols + col];
        }
        d_P[row*P_cols + col] = pValue;
    }
}

/*
GPU matrix multiplication using shared memory
*/
__global__ void matrixMultiplicationGPU_shared(float* d_M, float* d_N, float* d_P, int common_dim, int M_rows, int P_cols) {
    __shared__ float Mds[TILEWIDTH][TILEWIDTH];
    __shared__ float Nds[TILEWIDTH][TILEWIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILEWIDTH + ty;
    int col = bx * TILEWIDTH + tx;

    float p_value = 0;

    for (int m = 0; m < (common_dim + TILEWIDTH - 1) / TILEWIDTH; ++m) {
        int tiledCol = m * TILEWIDTH + tx;
        int tiledRow = m * TILEWIDTH + ty;
        
        if (row < M_rows && tiledCol < common_dim) {
            Mds[ty][tx] = d_M[row * common_dim + tiledCol]; 
        } else {
            Mds[ty][tx] = 0.0f;
        }
        
        if (tiledRow < common_dim && col < P_cols) {
            Nds[ty][tx] = d_N[tiledRow * P_cols + col]; 
        } else {
            Nds[ty][tx] = 0.0f;
        }
        
        __syncthreads();

        for (int k = 0; k < TILEWIDTH; ++k) {
            p_value += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }        
    if (row < M_rows && col < P_cols) {
        d_P[row * P_cols + col] = p_value;
    }
}

/*
GPU wrapper function to allocate memory, copy data, launch kernels, and measure time
*/
void gpu_wrapper(int m_width, int n_height, int common_dim, double* avg_time, float* gpu_result, float* host_M, float* host_N, bool shared_memory = true) {
    int p_width = n_height;   // number of cols in P
    int p_height = m_width;   // number of rows in P

    size_t m_bytes = (size_t)m_width * common_dim * sizeof(float);
    size_t n_bytes = (size_t)common_dim * n_height * sizeof(float);
    size_t p_bytes = (size_t)p_width * p_height * sizeof(float);

    float* gpu_M;
    float* gpu_N;
    float* gpu_P;

    cudaMalloc((void**)&gpu_M, m_bytes);
    cudaMalloc((void**)&gpu_N, n_bytes);
    cudaMalloc((void**)&gpu_P, p_bytes);

    // host_M and host_N are provided by the caller (initialized once in run_tests)
    cudaMemcpy(gpu_M, host_M, m_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_N, host_N, n_bytes, cudaMemcpyHostToDevice);

    double start = cpuSecond();
    if (shared_memory) {
        for (int i = 0; i < GPU_NUM_ITERATIONS; i++) {
            dim3 block(TILEWIDTH, TILEWIDTH);
            dim3 grid((p_width + TILEWIDTH - 1) / TILEWIDTH, (p_height + TILEWIDTH - 1) / TILEWIDTH);
            matrixMultiplicationGPU_shared<<<grid, block>>>(gpu_M, gpu_N, gpu_P, common_dim, m_width, p_width);        
            cudaDeviceSynchronize();
        }
    } else {
        for (int i = 0; i < GPU_NUM_ITERATIONS; i++) {
            dim3 block(16, 16);
            dim3 grid((int)ceil(p_width/(float)16), (int)ceil(p_height/(float)16), 1);
            matrixMultiplicationGPU_non_shared<<<grid, block>>>(gpu_M, gpu_N, gpu_P, common_dim, m_width, p_width);
            cudaDeviceSynchronize();
        }
    }
    double end = cpuSecond();
    *avg_time = (end - start) / GPU_NUM_ITERATIONS;
    cudaMemcpy(gpu_result, gpu_P, p_bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(gpu_M);
    cudaFree(gpu_N);
    cudaFree(gpu_P);
    
}

/*
CPU wrapper function to measure time for CPU implementations
*/
void cpu_wrapper(int m_width, int n_height, int common_dim, double* avg_time, float* cpu_result, float* host_M, float* host_N, bool shared_memory = true) {
    int p_width = n_height;   // number of cols in P

    // host_M and host_N are provided by the caller (initialized once in run_tests)
    double start = cpuSecond();
    if (shared_memory) {
        for (int i = 0; i < CPU_NUM_ITERATIONS; i++) {
            matrixMultiplicationCPU_shared(host_M, host_N, cpu_result, m_width, common_dim, p_width);
        }
    } else {
        for (int i = 0; i < CPU_NUM_ITERATIONS; i++) {
            matrixMultiplicationCPU_non_shared(host_M, host_N, cpu_result, m_width, common_dim, p_width);
        }
    }
    double end = cpuSecond();
    *avg_time = (end - start) / CPU_NUM_ITERATIONS;
}


/*
Function to run different tests based on the task number (main logic)
*/
void run_tests(int task) {
    switch(task) {
        case 2: { // 2) Compare execution time for versions with/without shared memor
            printf("Running Task 2: Compare shared vs non-shared memory implementations\n");
            // first running test on CPU
            int m_width = 10000;
            int n_height = 20000;
            int common_dim = 5000;
            float* cpu_result_shared = (float*) malloc((size_t)m_width * n_height * sizeof(float));
            float* cpu_result_non_shared = (float*) malloc((size_t)m_width * n_height * sizeof(float));
            double cpu_time_shared, cpu_time_non_shared;
            printf("shared vs non-shared memory CPU matrix multiplication:\n");
            printf("running shared memory version...\n");
            float* host_M_cpu = (float*) malloc((size_t)m_width * common_dim * sizeof(float));
            float* host_N_cpu = (float*) malloc((size_t)common_dim * n_height * sizeof(float));
            initialData(host_M_cpu, m_width * common_dim);
            initialData(host_N_cpu, common_dim * n_height);
            cpu_wrapper(m_width, n_height, common_dim, &cpu_time_shared, cpu_result_shared, host_M_cpu, host_N_cpu, true);
            printf("running non-shared memory version...\n");
            cpu_wrapper(m_width, n_height, common_dim, &cpu_time_non_shared, cpu_result_non_shared, host_M_cpu, host_N_cpu, false);
            printf("CPU Shared Memory Time: %f s\n", cpu_time_shared);
            printf("CPU Non-Shared Memory Time: %f s\n", cpu_time_non_shared);
            checkResult(cpu_result_shared, cpu_result_non_shared, m_width * n_height);
            
            // running same test on GPU
            printf("\n");
            
            // resulting in M: 10000 x 5000, N: 5000 x 20000, P: 10000 x 20000
            float* gpu_result_shared = (float*) malloc((size_t)m_width * n_height * sizeof(float));
            float* gpu_result_non_shared = (float*) malloc((size_t)m_width * n_height * sizeof(float));
            double gpu_time_shared, gpu_time_non_shared;
            printf("shared vs non-shared memory GPU matrix multiplication:\n");
            printf("running shared memory version...\n");
            float* host_M_gpu = (float*) malloc((size_t)m_width * common_dim * sizeof(float));
            float* host_N_gpu = (float*) malloc((size_t)common_dim * n_height * sizeof(float));
            initialData(host_M_gpu, m_width * common_dim);
            initialData(host_N_gpu, common_dim * n_height);
            gpu_wrapper(m_width, n_height, common_dim, &gpu_time_shared, gpu_result_shared, host_M_gpu, host_N_gpu, true);
            printf("running non-shared memory version...\n");
            gpu_wrapper(m_width, n_height, common_dim, &gpu_time_non_shared, gpu_result_non_shared, host_M_gpu, host_N_gpu, false);
            printf("GPU Shared Memory Time: %f s\n", gpu_time_shared);
            printf("GPU Non-Shared Memory Time: %f s\n", gpu_time_non_shared);
            checkResult(gpu_result_shared, gpu_result_non_shared, m_width * n_height);
            free(host_M_gpu);
            free(host_N_gpu);
            free(gpu_result_shared);
            free(gpu_result_non_shared);
            free(host_M_cpu);
            free(host_N_cpu);
            free(cpu_result_shared);
            free(cpu_result_non_shared);
            break;
        }
        case 3: {
            // 3) Compare execution time for the host and device implementations
            printf("Running Task 3: Compare CPU vs GPU implementations\n");
             int m_width = 10000;
            int n_height = 20000;
            int common_dim = 5000;

            float* cpu_result = (float*) malloc((size_t)m_width * n_height * sizeof(float));
            float* gpu_result = (float*) malloc((size_t)m_width * n_height * sizeof(float));
            double cpu_time, gpu_time;
            printf("CPU vs GPU matrix multiplication (shared memory versions):\n");
            printf("running CPU version...\n");
            float* host_M = (float*) malloc((size_t)m_width * common_dim * sizeof(float));
            float* host_N = (float*) malloc((size_t)common_dim * n_height * sizeof(float));
            initialData(host_M, m_width * common_dim);
            initialData(host_N, common_dim * n_height);
            cpu_wrapper(m_width, n_height, common_dim, &cpu_time, cpu_result, host_M, host_N, true);
            printf("running GPU version...\n");
            gpu_wrapper(m_width, n_height, common_dim, &gpu_time, gpu_result, host_M, host_N, true);
            printf("CPU Time: %f s\n", cpu_time);
            printf("GPU Time: %f s\n", gpu_time);
            checkResult(cpu_result, gpu_result, m_width * n_height);
            free(host_M);
            free(host_N);
            free(cpu_result);
            free(gpu_result);
            break;
        }
        case 41: {
        /* 
        4.1: only TILE_WIDTH dependency. So only run the GPU version with different TILE_WIDTHs.
        For this the file has to be recompiled with different TILE_WIDTH values.
        */
            printf("Running Task 4.1: Analyze dependence of TILE_WIDTH\n");
            int m_width = 10000;
            int n_height = 20000;
            int common_dim = 5000;

            float* gpu_result = (float*) malloc((size_t)m_width * n_height * sizeof(float));
            double gpu_time;
            printf("GPU matrix multiplication (shared memory version) with TILE_WIDTH=%d:\n", TILEWIDTH);
            float* host_M = (float*) malloc((size_t)m_width * common_dim * sizeof(float));
            float* host_N = (float*) malloc((size_t)common_dim * n_height * sizeof(float));
            initialData(host_M, m_width * common_dim);
            initialData(host_N, common_dim * n_height); 
            gpu_wrapper(m_width, n_height, common_dim, &gpu_time, gpu_result, host_M, host_N, true);
            printf("GPU Time with TILE_WIDTH=%d: %f s\n", TILEWIDTH, gpu_time);
            free(host_M);
            free(host_N);
            free(gpu_result);   
            break;
        }
        case 42: {
        /*
        Study the dependence of the speedup factor of the GPU vs CPU execution on the matrix size.
        */
            printf("Running Task 4.2: Analyze speedup of matrix size\n");
            int matrix_sizes[5][2][2] = { {{100, 100}, {50, 50}}, {{500, 500}, {250, 250}}, {{1000, 1000}, {500, 500}}, 
                                            {{2000, 2000}, {1000, 1000}}, {{5000, 5000}, {2500, 2500}}};

            for (int i = 0; i < 5; i++) {
                int m_width = matrix_sizes[i][0][0];
                int n_height = matrix_sizes[i][0][1];
                int common_dim = matrix_sizes[i][1][0]; // square matrices
                int num_elements = m_width * n_height;
                double cpu_time, gpu_time;
                float* cpu_result = (float*) malloc((size_t)num_elements * sizeof(float));
                float* gpu_result = (float*) malloc((size_t)num_elements * sizeof(float));
                printf("Matrix size: M(%d x %d), N(%d x %d)\n", m_width, common_dim, common_dim, n_height);
                float* host_M = (float*) malloc((size_t)m_width * common_dim * sizeof(float));
                float* host_N = (float*) malloc((size_t)common_dim * n_height * sizeof(float));
                initialData(host_M, m_width * common_dim);
                initialData(host_N, common_dim * n_height);
                cpu_wrapper(m_width, n_height, common_dim, &cpu_time, cpu_result, host_M, host_N, true);
                gpu_wrapper(m_width, n_height, common_dim, &gpu_time, gpu_result, host_M, host_N, true);
                printf("CPU Time: %f s\n", cpu_time);
                printf("GPU Time: %f s\n", gpu_time);
                printf("Speedup (%d elements in result): (CPU time / GPU time): %f\n", num_elements, cpu_time / gpu_time);
                checkResult(cpu_result, gpu_result, num_elements);
                free(host_M);
                free(host_N);
                free(cpu_result);
                free(gpu_result);
            }

            break;
        }

    }
}
int main() {
    // run_tests(2);
    // run_tests(3);
    // run_tests(42);
    run_tests(41);
}