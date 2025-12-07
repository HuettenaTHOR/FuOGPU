#include <cuda.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
/*
For this Ben Olschar, 108021211678 and Frederik Hüttemann, 108021215247 cooperated with eachother. 
We did implement the code on our own, but agreed to using one version. The discussion and results are made in cooperation.
*/

// define some constants
#define CPU_NUM_ITERATIONS 10
#define GPU_NUM_ITERATIONS 1000

#define FULL_HD_WIDTH 1920
#define FULL_HD_HEIGHT 1080
#define TARGET_4K_WIDTH 3840
#define TARGET_4K_HEIGHT 2160
#define FILTER_WIDTH 19

__constant__ float kernel_const_gpu[FILTER_WIDTH * FILTER_WIDTH];

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
    double eps = 1.0E-3;
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

// CPU function to upscale an image using bilinear interpolation
void upscaleCPU(float *input, float *output, float* kernel) {
    // Allocate intermediate buffer for upscaled image
    float *intermediate = (float *)malloc(TARGET_4K_WIDTH * TARGET_4K_HEIGHT * sizeof(float));

    // run billinear interpolation
    for (int y = 0; y < TARGET_4K_HEIGHT; y++) {
        for (int x = 0; x < TARGET_4K_WIDTH; x++) {
            float in_out_ratio = (float)FULL_HD_WIDTH / (float)TARGET_4K_WIDTH; // should be 0.5 for 1920->3840
            float base_x = ((float)x) * in_out_ratio;
            float base_y = ((float)y) * in_out_ratio;
            int gxi = (int)base_x;
            int base_index_x_0 = (int)(base_x);
            int base_index_x_1;
            if (base_x + 1 >= FULL_HD_WIDTH) {
                base_index_x_1 = base_index_x_0;
            } else {
                base_index_x_1 = base_index_x_0 + 1;
            }
            
            int base_index_y_0 = (int)(base_y);
            int base_index_y_1;
            if (base_y + 1 >= FULL_HD_HEIGHT) {
                base_index_y_1 = base_index_y_0;
            } else {
                base_index_y_1 = base_index_y_0 + 1;
            }
            float q00 = input[base_index_y_0 * FULL_HD_WIDTH + base_index_x_0];
            float q10 = input[base_index_y_0 * FULL_HD_WIDTH + base_index_x_1];
            float q01 = input[base_index_y_1 * FULL_HD_WIDTH + base_index_x_0];
            float q11 = input[base_index_y_1 * FULL_HD_WIDTH + base_index_x_1];
            
            intermediate[y * TARGET_4K_WIDTH + x] = q00*(1 - (base_x - gxi))*(1 - (base_y - (int)base_y)) +
                                      q10*(base_x - gxi)*(1 - (base_y - (int)base_y)) +
                                      q01*(1 - (base_x - gxi))*(base_y - (int)base_y) +
                                      q11*(base_x - gxi)*(base_y - (int)base_y);
        }            
    }

    // run convolution with 19x19 filter
    for (int y = 0; y < TARGET_4K_HEIGHT; y++) {
        for (int x = 0; x < TARGET_4K_WIDTH; x++) {
            float pValue = 0;
            int radius = FILTER_WIDTH / 2;
            
            // 2D convolution: iterate over 19x19 neighborhood
            for (int ky = -radius; ky <= radius; ky++) {
                for (int kx = -radius; kx <= radius; kx++) {
                    int srcY = y + ky;
                    int srcX = x + kx;
                    
                    // Check bounds and apply zero-padding for ghost pixels
                    if (srcY >= 0 && srcY < TARGET_4K_HEIGHT && srcX >= 0 && srcX < TARGET_4K_WIDTH) {
                        int src_index = srcY * TARGET_4K_WIDTH + srcX;
                        int kernel_index = (ky + radius) * FILTER_WIDTH + (kx + radius);
                        pValue += intermediate[src_index] * kernel[kernel_index];
                    }
                    // else: ghost pixel, contributes 0
                }
            }
            output[y * TARGET_4K_WIDTH + x] = pValue;
        }
    }
    
    // Free intermediate buffer
    free(intermediate);
}

__global__ void upscaleGPU_global(float *input, float *output) {
    // GPU bilinear interpolation
    int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
    int gpu_index = xIndex + yIndex*TARGET_4K_WIDTH;
    float ratio = (float)FULL_HD_WIDTH / (float)TARGET_4K_WIDTH; // should be 0.5 for 1920->3840

    float base_x = ((float)xIndex) * ratio;
    float base_y = ((float)yIndex) * ratio;

    int gxi = (int)base_x;
    int base_index_x_0 = (int)(base_x);
    int base_index_x_1;
    if (base_x + 1 >= FULL_HD_WIDTH) {
        base_index_x_1 = base_index_x_0;
    } else {
        base_index_x_1 = base_index_x_0 + 1;
    }

    int base_index_y_0 = (int)(base_y);
    int base_index_y_1;
    if (base_y + 1 >= FULL_HD_HEIGHT) {
        base_index_y_1 = base_index_y_0;
    } else {
        base_index_y_1 = base_index_y_0 + 1;
    }
    float q00 = input[base_index_y_0 * FULL_HD_WIDTH + base_index_x_0];
    float q10 = input[base_index_y_0 * FULL_HD_WIDTH + base_index_x_1];
    float q01 = input[base_index_y_1 * FULL_HD_WIDTH + base_index_x_0];
    float q11 = input[base_index_y_1 * FULL_HD_WIDTH + base_index_x_1];
    output[gpu_index] = q00*(1 - (base_x - gxi))*(1 - (base_y - (int)base_y)) +
        q10*(base_x - gxi)*(1 - (base_y - (int)base_y)) +
        q01*(1 - (base_x - gxi))*(base_y - (int)base_y) +
        q11*(base_x - gxi)*(base_y - (int)base_y);
}

__global__ void convolveGPU_global(float *input, float *output, float* kernel) {
    // GPU convolution with 19x19 filter
    int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
    
    if (xIndex >= TARGET_4K_WIDTH || yIndex >= TARGET_4K_HEIGHT) return;
    
    int gpu_index = xIndex + yIndex*TARGET_4K_WIDTH;
    float Pvalue = 0;
    int radius = FILTER_WIDTH / 2;
    
    // 2D convolution: iterate over 19x19 neighborhood
    for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
            int srcY = yIndex + ky;
            int srcX = xIndex + kx;
            
            // Check bounds and apply zero-padding for ghost pixels
            if (srcY >= 0 && srcY < TARGET_4K_HEIGHT && srcX >= 0 && srcX < TARGET_4K_WIDTH) {
                int src_index = srcY * TARGET_4K_WIDTH + srcX;
                int kernel_index = (ky + radius) * FILTER_WIDTH + (kx + radius);
                Pvalue += input[src_index] * kernel[kernel_index];
            }
            // else: ghost pixel, contributes 0
        }
    }
    output[gpu_index] = Pvalue;
}

__global__ void convolveGPU_constant(float *input, float *output) {
    // GPU convolution with 19x19 filter using constant memory
    int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
    
    if (xIndex >= TARGET_4K_WIDTH || yIndex >= TARGET_4K_HEIGHT) return;
    
    int gpu_index = xIndex + yIndex*TARGET_4K_WIDTH;
    float Pvalue = 0;
    int radius = FILTER_WIDTH / 2;
    
    // 2D convolution: iterate over 19x19 neighborhood
    for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
            int srcY = yIndex + ky;
            int srcX = xIndex + kx;
            
            // Check bounds and apply zero-padding for ghost pixels
            if (srcY >= 0 && srcY < TARGET_4K_HEIGHT && srcX >= 0 && srcX < TARGET_4K_WIDTH) {
                int src_index = srcY * TARGET_4K_WIDTH + srcX;
                int kernel_index = (ky + radius) * FILTER_WIDTH + (kx + radius);
                Pvalue += input[src_index] * kernel_const_gpu[kernel_index];
            }
            // else: ghost pixel, contributes 0
        }
    }
    output[gpu_index] = Pvalue;
}


__global__ void upscaleGPU_texture(cudaTextureObject_t input, float *output) {
    int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
    int gpu_index = xIndex + yIndex*TARGET_4K_WIDTH;
    float ratio = (float)FULL_HD_WIDTH / (float)TARGET_4K_WIDTH; // should be 0.5 for 1920->3840

    float base_x = ((float)xIndex) * ratio;
    float base_y = ((float)yIndex) * ratio;

    int gxi = (int)base_x;
    int base_index_x_0 = (int)(base_x);
    int base_index_x_1;
    if (base_x + 1 >= FULL_HD_WIDTH) {
        base_index_x_1 = base_index_x_0;
    } else {
        base_index_x_1 = base_index_x_0 + 1;
    }

    int base_index_y_0 = (int)(base_y);
    int base_index_y_1;
    if (base_y + 1 >= FULL_HD_HEIGHT) {
        base_index_y_1 = base_index_y_0;
    } else {
        base_index_y_1 = base_index_y_0 + 1;
    }
    float q00 = tex2D<float>(input, base_index_x_0, base_index_y_0);
    float q10 = tex2D<float>(input, base_index_x_1, base_index_y_0);
    float q01 = tex2D<float>(input, base_index_x_0, base_index_y_1);
    float q11 = tex2D<float>(input, base_index_x_1, base_index_y_1);

    output[gpu_index] = q00*(1 - (base_x - gxi))*(1 - (base_y - (int)base_y)) +
        q10*(base_x - gxi)*(1 - (base_y - (int)base_y)) +
        q01*(1 - (base_x - gxi))*(base_y - (int)base_y) +
        q11*(base_x - gxi)*(base_y - (int)base_y);
}


void run_tests(int task) {
    switch(task) {
        case 1: {
            printf("Running Task 1: CPU Upscaling + Convolution\n");
            int inWidth = FULL_HD_WIDTH;
            int inHeight = FULL_HD_HEIGHT;
            int outWidth = TARGET_4K_WIDTH;
            int outHeight = TARGET_4K_HEIGHT;

            int kernelSize = FILTER_WIDTH * FILTER_WIDTH * sizeof(float);
            float *kernel_cpu = (float *)malloc(kernelSize);

            // initialize a simple averaging kernel for demonstration
            for (int i = 0; i < FILTER_WIDTH * FILTER_WIDTH; i++) {
                kernel_cpu[i] = 1.0f / (FILTER_WIDTH * FILTER_WIDTH);
            }

            size_t inSize = inWidth * inHeight * sizeof(float);
            size_t outSize = outWidth * outHeight * sizeof(float);

            // allocate host memory
            float *input_cpu = (float *)malloc(inSize);
            float *output_cpu = (float *)malloc(outSize);

            // initialize input data
            initialData(input_cpu, inWidth * inHeight);

            // CPU Upscaling
            double cpuStart = cpuSecond();
            for (int i = 0; i < CPU_NUM_ITERATIONS; i++) {
                upscaleCPU(input_cpu, output_cpu, kernel_cpu);
            }
            double cpuEnd = cpuSecond();
            printf("CPU Upscaling + Convolution Time: %f seconds\n", (cpuEnd - cpuStart) / CPU_NUM_ITERATIONS);

            // Free host memory
            free(input_cpu);
            free(output_cpu);
            free(kernel_cpu);
            break;
        }
        case 2: {
            printf("Running Task 2: GPU Upscaling + Convolution\n");
            int inWidth = FULL_HD_WIDTH;
            int inHeight = FULL_HD_HEIGHT;
            int outWidth = TARGET_4K_WIDTH;
            int outHeight = TARGET_4K_HEIGHT;

            int kernelSize = FILTER_WIDTH * FILTER_WIDTH * sizeof(float);
            float *kernel_cpu = (float*)malloc(kernelSize);

            // initialize a simple averaging kernel for demonstration
            for (int i = 0; i < FILTER_WIDTH * FILTER_WIDTH; i++) {
                kernel_cpu[i] = 1.0f / (FILTER_WIDTH * FILTER_WIDTH);
            }

            size_t inSize = inWidth * inHeight * sizeof(float);
            size_t outSize = outWidth * outHeight * sizeof(float);

            // allocate host memory
            float *input_cpu = (float*)malloc(inSize);
            float *output_cpu = (float*)malloc(outSize);
            float *output_gpu = (float*)malloc(outSize);
            
            float *gpu_in;
            float *gpu_intermediate;
            float *gpu_out; 
            float *gpu_kernel;
            cudaMalloc((void**)&gpu_in, inSize);
            cudaMalloc((void**)&gpu_intermediate, outSize);
            cudaMalloc((void**)&gpu_out, outSize);
            cudaMalloc((void**)&gpu_kernel, kernelSize);

            // initialize input data
            initialData(input_cpu, inWidth * inHeight);

            // push data to GPU
            cudaMemcpy(gpu_in, input_cpu, inSize, cudaMemcpyHostToDevice);
            cudaMemcpy(gpu_kernel, kernel_cpu, kernelSize, cudaMemcpyHostToDevice);

            // run reference CPU implementation
            upscaleCPU(input_cpu, output_cpu, kernel_cpu);

            int block_size = 32;
            dim3 block (block_size, block_size);
            dim3 grid(ceil(outWidth/(float)block_size), ceil(outHeight/(float)block_size), 1);

            // warm up
            upscaleGPU_global<<<grid, block>>>(gpu_in, gpu_intermediate);
            cudaDeviceSynchronize();
            convolveGPU_global<<<grid, block>>>(gpu_intermediate, gpu_out, gpu_kernel);
            cudaDeviceSynchronize();

            double gpu_start = cpuSecond();
            for(int i = 0; i < GPU_NUM_ITERATIONS; i++) {
                upscaleGPU_global<<<grid, block>>>(gpu_in, gpu_intermediate);
                cudaDeviceSynchronize();
                convolveGPU_global<<<grid, block>>>(gpu_intermediate, gpu_out, gpu_kernel);
                cudaDeviceSynchronize();
            }
            double gpu_end = cpuSecond();
            printf("GPU Upscaling + Convolution (global memory) Time: %f ms \n", (gpu_end - gpu_start) / GPU_NUM_ITERATIONS * 1000);

            cudaMemcpy(output_gpu, gpu_out, outSize, cudaMemcpyDeviceToHost);

            cudaFree(gpu_in);
            cudaFree(gpu_intermediate);
            cudaFree(gpu_out);
            cudaFree(gpu_kernel);

            checkResult(output_cpu, output_gpu, outWidth*outHeight);

            free(kernel_cpu);
            free(input_cpu);
            free(output_cpu);
            free(output_gpu);

            break;
        }
        case 3: {
            printf("Running Task 3: GPU Upscaling + Convolution with Constant Memory\n");
            int inWidth = FULL_HD_WIDTH;
            int inHeight = FULL_HD_HEIGHT;
            int outWidth = TARGET_4K_WIDTH;
            int outHeight = TARGET_4K_HEIGHT;

            int kernelSize = FILTER_WIDTH * FILTER_WIDTH * sizeof(float);
            float *kernel_cpu = (float*)malloc(kernelSize);            

            // initialize a simple averaging kernel for demonstration
            for (int i = 0; i < FILTER_WIDTH * FILTER_WIDTH; i++) {
                kernel_cpu[i] = 1.0f / (FILTER_WIDTH * FILTER_WIDTH);
            }
            // Copy kernel to constant memory
            cudaMemcpyToSymbol(kernel_const_gpu, kernel_cpu, kernelSize);

            size_t inSize = inWidth * inHeight * sizeof(float);
            size_t outSize = outWidth * outHeight * sizeof(float);

            // allocate host memory
            float *input_cpu = (float*)malloc(inSize);
            float *output_cpu = (float*)malloc(outSize);
            float *output_gpu = (float*)malloc(outSize);
            
            float *gpu_in;
            float *gpu_intermediate;
            float *gpu_out; 
            cudaMalloc((void**)&gpu_in, inSize);
            cudaMalloc((void**)&gpu_intermediate, outSize);
            cudaMalloc((void**)&gpu_out, outSize);

            // initialize input data
            initialData(input_cpu, inWidth * inHeight);

            // push data to GPU
            cudaMemcpy(gpu_in, input_cpu, inSize, cudaMemcpyHostToDevice);

            // run reference CPU implementation
            upscaleCPU(input_cpu, output_cpu, kernel_cpu);

            int block_size = 32;
            dim3 block (block_size, block_size);
            dim3 grid(ceil(outWidth/(float)block_size), ceil(outHeight/(float)block_size), 1);

            // warm up
            upscaleGPU_global<<<grid, block>>>(gpu_in, gpu_intermediate);
            cudaDeviceSynchronize();
            convolveGPU_constant<<<grid, block>>>(gpu_intermediate, gpu_out);
            cudaDeviceSynchronize();

            double gpu_start = cpuSecond();
            for(int i = 0; i < GPU_NUM_ITERATIONS; i++) {
                upscaleGPU_global<<<grid, block>>>(gpu_in, gpu_intermediate);
                cudaDeviceSynchronize();
                convolveGPU_constant<<<grid, block>>>(gpu_intermediate, gpu_out);
                cudaDeviceSynchronize();
            }
            double gpu_end = cpuSecond();
            printf("GPU Upscaling + Convolution (constant memory) Time: %f ms \n", (gpu_end - gpu_start) / GPU_NUM_ITERATIONS * 1000);

            cudaMemcpy(output_gpu, gpu_out, outSize, cudaMemcpyDeviceToHost);

            cudaFree(gpu_in);
            cudaFree(gpu_intermediate);
            cudaFree(gpu_out);

            checkResult(output_cpu, output_gpu, outWidth*outHeight);

            free(kernel_cpu);
            free(input_cpu);
            free(output_cpu);
            free(output_gpu);

            break;
        }
        case 4: {
            printf("Running Task 4: GPU Upscaling with Texture Memory + Convolution with Constant Memory\n");
            
            int inWidth = FULL_HD_WIDTH;
            int inHeight = FULL_HD_HEIGHT;
            int outWidth = TARGET_4K_WIDTH;
            int outHeight = TARGET_4K_HEIGHT;

            int kernelSize = FILTER_WIDTH * FILTER_WIDTH * sizeof(float);
            float *kernel_cpu = (float*)malloc(kernelSize);            

            // initialize a simple averaging kernel for demonstration
            for (int i = 0; i < FILTER_WIDTH * FILTER_WIDTH; i++) {
                kernel_cpu[i] = 1.0f / (FILTER_WIDTH * FILTER_WIDTH);
            }
            // Copy kernel to constant memory
            cudaMemcpyToSymbol(kernel_const_gpu, kernel_cpu, kernelSize);

            size_t inSize = inWidth * inHeight * sizeof(float);
            size_t outSize = outWidth * outHeight * sizeof(float);

            // allocate host memory
            float *input_cpu = (float*)malloc(inSize);
            float *output_cpu = (float*)malloc(outSize);
            float *output_gpu = (float*)malloc(outSize);
            
            float *gpu_intermediate;
            float *gpu_out; 
            cudaMalloc((void**)&gpu_intermediate, outSize);
            cudaMalloc((void**)&gpu_out, outSize);

            // initialize input data
            initialData(input_cpu, inWidth * inHeight);

            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
            cudaArray* cuArray;
            cudaMallocArray(&cuArray, &channelDesc, inWidth, inHeight);
            cudaMemcpy2DToArray(cuArray, 0, 0, input_cpu, inWidth * sizeof(float), inWidth * sizeof(float), inHeight, cudaMemcpyHostToDevice);

            // create texture object
            struct cudaResourceDesc resDesc;
            memset(&resDesc, 0, sizeof(resDesc));
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = cuArray;

            struct cudaTextureDesc texDesc;
            memset(&texDesc, 0, sizeof(texDesc));
            texDesc.addressMode[0] = cudaAddressModeClamp;
            texDesc.addressMode[1] = cudaAddressModeClamp;
            texDesc.filterMode = cudaFilterModePoint;
            texDesc.readMode = cudaReadModeElementType;
            texDesc.normalizedCoords = 0;

            cudaTextureObject_t texObj = 0;
            cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

            // run reference CPU implementation
            upscaleCPU(input_cpu, output_cpu, kernel_cpu);

            int block_size = 32;
            dim3 block (block_size, block_size);
            dim3 grid(ceil(outWidth/(float)block_size), ceil(outHeight/(float)block_size), 1);

            // warm up
            upscaleGPU_texture<<<grid, block>>>(texObj, gpu_intermediate);
            cudaDeviceSynchronize();
            convolveGPU_constant<<<grid, block>>>(gpu_intermediate, gpu_out);
            cudaDeviceSynchronize();

            double gpu_start = cpuSecond();
            for(int i = 0; i < GPU_NUM_ITERATIONS; i++) {
                upscaleGPU_texture<<<grid, block>>>(texObj, gpu_intermediate);
                cudaDeviceSynchronize();
                convolveGPU_constant<<<grid, block>>>(gpu_intermediate, gpu_out);
                cudaDeviceSynchronize();
            }
            double gpu_end = cpuSecond();
            printf("GPU Upscaling + Convolution (constant memory) Time: %f ms \n", (gpu_end - gpu_start) / GPU_NUM_ITERATIONS * 1000);

            cudaMemcpy(output_gpu, gpu_out, outSize, cudaMemcpyDeviceToHost);

            
            cudaDestroyTextureObject(texObj);
            cudaFreeArray(cuArray);

            cudaFree(gpu_intermediate);
            cudaFree(gpu_out);

            checkResult(output_cpu, output_gpu, outWidth*outHeight);

            free(kernel_cpu);
            free(input_cpu);
            free(output_cpu);
            free(output_gpu);
            break;
        }
        default:
            printf("Invalid task number.\n");
    }
}

int main() {
    // run_tests(1);
    // run_tests(2);
    run_tests(3);
    run_tests(4);
    return 0;
}