//#include <stdio.h>
#include <cstdio>

__global__ void HelloWorld(void)
{
    printf("Hello World from GPU \n");
}

int main(void)
{
    HelloWorld <<<1, 10>>>();
    cudaDeviceReset();
    return 0;
}