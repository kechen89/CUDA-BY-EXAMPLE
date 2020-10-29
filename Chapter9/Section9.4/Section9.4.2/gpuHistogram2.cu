#include "../../../common/book.h"
#include <iostream>

#define SIZE (100 * 1024 * 1024)

__global__ void histo_kernel(unsigned char *buffer, long size, unsigned int *histo){
    __shared__ unsigned int temp[256];
    temp[threadIdx.x] = 0;
    __syncthreads();
 
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (i < size) {
        atomicAdd(&(temp[buffer[i]]),1);
        i += stride;
    }

    __syncthreads();
    atomicAdd(&(histo[threadIdx.x]), temp[threadIdx.x]);
}

int main(void){
    unsigned char *buffer = (unsigned char*)big_random_block(SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    // allocate memory on the GPU for the file's data
    unsigned char *dev_buffer;
    unsigned int *dev_histo;
   
    cudaMalloc((void**)&dev_buffer, SIZE);
    cudaMemcpy(dev_buffer, buffer, SIZE, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&dev_histo, 256 * sizeof(long));
    cudaMemset(dev_histo, 0, 256 *sizeof(int));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int blocks = prop.multiProcessorCount;
   
    histo_kernel<<<blocks * 2, 256>>>(dev_buffer, SIZE, dev_histo);
   
    unsigned int histo[256];
    cudaMemcpy(histo, dev_histo, 256 * sizeof(int), cudaMemcpyDeviceToHost);
 
    // get stop time, and display the timing results
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Time to generate: " << elapsedTime << std::endl;

    long histoCount = 0;
    for (int i = 0; i < 256; i++)
        histoCount += histo[i];
    
    std::cout << "Histogram Sum: " << histoCount << std::endl;

    // verify that we have the same counts via CPU
    for (int i = 0; i < SIZE; i++)
        histo[buffer[i]]--;

    for (int i = 0; i < 256; i++)
        if (histo[i] != 0)
            std::cout << "Failure at " << i << "!"  << std::endl;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(dev_histo);
    cudaFree(dev_buffer);
    free(buffer);
    return 0;
}
