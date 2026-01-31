#include <cuda.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>

#define BLOCK_SIZE 16
#define NUM_PROCESSING_IMAGES 100
#define TARGET_TIME_MS 8.0f // Target time in milliseconds for each part of the execution
int image_size = 10; // to be determined in Task 0
int num_reps = 10; // to be determined in Task 0

void initialData(float *ip, int size_m, int size_n){
    
    time_t t;
    srand((unsigned) time(&t));

    for (int i=0; i<size_m; i++){
        for (int j=0; j<size_n; j++){
            ip[i*size_n + j] = (float)(rand() & 0xFF) / 10.0f;
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

// some kernel functionality where the execution time can be adjusted via num_repetitions
__global__ void process_kernel(float *input_image, float *output_image, int size_m, int size_n, int num_repetitions) {
    
    for(int rep = 0; rep < num_repetitions; rep++) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < size_m && col < size_n) {
            // Simple processing: just copy input to output
            output_image[row * size_n + col] = input_image[row * size_n + col];
        }
    }
}

void run_test(int task) {
    switch(task) {
        case 0: {
            printf("Running Task 0: finding optimal hyperparamters so \"host-to-device\", \"kernel execution\", and \"device-to-host\" each take approximately %.2f ms\n", TARGET_TIME_MS);
            
            printf("Finding optimal image size for host-to-device transfer...\n");
            for (int size = 1024; size <= 16384; size += 512) {
                float *h_image = (float *)malloc(size * size * sizeof(float));
                float *d_image;
                initialData(h_image, size, size);
                cudaMalloc((void **)&d_image, size * size * sizeof(float));

                double start = cpuSecond();
                cudaMemcpy(d_image, h_image, size * size * sizeof(float), cudaMemcpyHostToDevice);
                cudaDeviceSynchronize();
                double elapsed = (cpuSecond() - start) * 1000.0f; // in ms
                printf("Image size: %d x %d, Host-to-Device time: %.2f ms\n", size, size, elapsed);
                double copy_back_start = cpuSecond();
                cudaMemcpy(h_image, d_image, size * size * sizeof(float), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                double copy_back_elapsed = (cpuSecond() - copy_back_start) * 1000.0f; // in ms
                printf("Image size: %d x %d, Device-to-Host time: %.2f ms\n\n", size, size, copy_back_elapsed);
                cudaFree(d_image);
                free(h_image);

                if (elapsed >= TARGET_TIME_MS) {
                    printf("Optimal image size for host-to-device transfer: %d x %d (%.2f ms)\n", size, size, elapsed);
                    image_size = size;
                    break;
                }
            }
            printf("Finding optimal kernel execution repetition time...\n");
            float *host_input_image = (float *)malloc(image_size * image_size * sizeof(float));
            float *device_input_image, *device_output_image;
            initialData(host_input_image, image_size, image_size);
            cudaMalloc((void **)&device_input_image, image_size * image_size * sizeof(float));
            cudaMalloc((void **)&device_output_image, image_size * image_size * sizeof(float));
            cudaMemcpy(device_input_image, host_input_image, image_size * image_size * sizeof(float), cudaMemcpyHostToDevice);    
            dim3 block(BLOCK_SIZE, BLOCK_SIZE);
            dim3 grid((image_size + block.x - 1) / block.x, (image_size + block.y - 1) / block.y);
            for (int repetitions = 1; repetitions <= 1000; repetitions++) {
                double start = cpuSecond();
                process_kernel<<<grid, block>>>(device_input_image, device_output_image, image_size, image_size, repetitions);
                cudaDeviceSynchronize();
                double elapsed = (cpuSecond() - start) * 1000.0f; // in ms
                if (repetitions % 10 == 0) printf("Repetitions: %d, Kernel execution time: %.2f ms\n", repetitions, elapsed);
                if (elapsed >= TARGET_TIME_MS) {
                    num_reps = repetitions;
                    printf("Optimal repetitions for kernel execution: %d (%.2f ms)\n", repetitions, elapsed);
                    break;
                }
            }
            cudaFree(device_input_image);
            cudaFree(device_output_image);
            free(host_input_image);

            printf("With the determined image size of %d x %d, host-to-device transfer, kernel execution, and device-to-host transfer should each take approximately %.2f ms.\n", image_size, image_size, TARGET_TIME_MS);
            break;
        }
        case 1: {
            printf("Running %d images of size %d x %d through the kernel with %d repetitions each without streaming...\n", NUM_PROCESSING_IMAGES, image_size, image_size, num_reps);
            size_t image_bytes = (size_t)image_size * image_size * sizeof(float);
            size_t total_bytes = NUM_PROCESSING_IMAGES * image_bytes;
            float *host_input_images = (float *)malloc(total_bytes);
            float *host_output_images = (float *)malloc(total_bytes);
            if (!host_input_images || !host_output_images) {
                printf("Host memory allocation failed!\n");
                break;
            }
            float *device_input_images, *device_output_images;
            for (int i = 0; i < NUM_PROCESSING_IMAGES; i++) {
                initialData(&host_input_images[i * image_size * image_size], image_size, image_size);
            }
            cudaError_t err1 = cudaMalloc((void **)&device_input_images, total_bytes);
            cudaError_t err2 = cudaMalloc((void **)&device_output_images, total_bytes);
            if (err1 != cudaSuccess || err2 != cudaSuccess) {
                printf("CUDA memory allocation failed: %s, %s\n", cudaGetErrorString(err1), cudaGetErrorString(err2));
                free(host_input_images);
                free(host_output_images);
                break;
            }
            dim3 block(BLOCK_SIZE, BLOCK_SIZE);
            dim3 grid((image_size + block.x - 1) / block.x, (image_size + block.y - 1) / block.y);
            double start = cpuSecond();
            for (int i = 0; i < NUM_PROCESSING_IMAGES; i++) {
                cudaMemcpy(&device_input_images[i * image_size * image_size], &host_input_images[i * image_size * image_size], image_bytes, cudaMemcpyHostToDevice);
            }
            for (int i = 0; i < NUM_PROCESSING_IMAGES; i++) {
                process_kernel<<<grid, block>>>(&device_input_images[i * image_size * image_size], &device_output_images[i * image_size * image_size], image_size, image_size, num_reps);
            }
            for (int i = 0; i < NUM_PROCESSING_IMAGES; i++) {
                cudaMemcpy(&host_output_images[i * image_size * image_size], &device_output_images[i * image_size * image_size], image_bytes, cudaMemcpyDeviceToHost);
            }
            cudaDeviceSynchronize();
            double elapsed = (cpuSecond() - start) * 1000.0f;
            printf("Total time without streaming for %d images: %.2f ms\n\n\n\n", NUM_PROCESSING_IMAGES, elapsed);
            cudaFree(device_input_images);
            cudaFree(device_output_images);
            free(host_input_images);
            free(host_output_images);
            break;
        }
        
    case 2:
        {
            printf("Running %d images of size %d x %d through the kernel with %d repetitions each with streaming...\n", NUM_PROCESSING_IMAGES, image_size, image_size, num_reps);
            size_t image_bytes = (size_t)image_size * image_size * sizeof(float);
            size_t total_bytes = NUM_PROCESSING_IMAGES * image_bytes;
            float *host_input_images, *host_output_images;
            // usage of pinned memory for better performance with streams as this uses the shared pageable memory area
            cudaMallocHost((void **)&host_input_images, total_bytes);
            cudaMallocHost((void **)&host_output_images, total_bytes);
            
            for (int i = 0; i < NUM_PROCESSING_IMAGES; i++) {
                initialData(&host_input_images[i * image_size * image_size], image_size, image_size);
            }
            cudaStream_t streams[NUM_PROCESSING_IMAGES];
            for (int i = 0; i < NUM_PROCESSING_IMAGES; i++) {
                cudaStreamCreate(&streams[i]);
            }
            float *device_input_images, *device_output_images;
            cudaMalloc((void **)&device_input_images, total_bytes);
            cudaMalloc((void **)&device_output_images, total_bytes);
            dim3 block(BLOCK_SIZE, BLOCK_SIZE);
            dim3 grid((image_size + block.x - 1) / block.x, (image_size + block.y - 1) / block.y);
            double start = cpuSecond();
            for (int i = 0; i < NUM_PROCESSING_IMAGES; i++) {
                cudaMemcpyAsync(&device_input_images[i * image_size * image_size], &host_input_images[i * image_size * image_size], image_bytes, cudaMemcpyHostToDevice, streams[i]);
                process_kernel<<<grid, block, 0, streams[i]>>>(&device_input_images[i * image_size * image_size], &device_output_images[i * image_size * image_size], image_size, image_size, num_reps);
                cudaMemcpyAsync(&host_output_images[i * image_size * image_size], &device_output_images[i * image_size * image_size], image_bytes, cudaMemcpyDeviceToHost, streams[i]);
            }
            for (int i = 0; i < NUM_PROCESSING_IMAGES; i++) {
                cudaStreamSynchronize(streams[i]);  
            }
            double elapsed = (cpuSecond() - start) * 1000.0f;
            printf("Total time with streaming for %d images: %.2f ms\n", NUM_PROCESSING_IMAGES, elapsed);
            for (int i = 0; i < NUM_PROCESSING_IMAGES; i++) {
                cudaStreamDestroy(streams[i]);
            }
            cudaFree(device_input_images);
            cudaFree(device_output_images);
            // free pinned memory
            cudaFreeHost(host_input_images);
            cudaFreeHost(host_output_images);
            break;
        }
        default: {
            printf("Invalid task number. Please choose a valid task.\n");
            break;
        }

    }
}

int main() {
    run_test(0);
    run_test(1);
    run_test(2);
    return 0;
}