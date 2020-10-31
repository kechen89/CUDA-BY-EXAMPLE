#include <iostream>

#define N (1024*1024)
#define FULL_DATA_SIZE (N*20)

__global__ void kernel(int *a, int *b, int *c){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        int idx1 = (idx + 1) % 256;
        int idx2 = (idx + 2) % 256;
        float as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
        float bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
        c[idx] = (as + bs)/2;
    }
}

int main(){
    cudaDeviceProp prop;
    int whichDevice;
    cudaGetDevice(&whichDevice);
    cudaGetDeviceProperties(&prop, whichDevice);
    if (!prop.deviceOverlap) {
        std::cout << "Device will not handle overlaps, so no speed up from streams" << std::endl;

        return 0;
    }

    cudaEvent_t start, stop;
    float elapsedTime;

    // start the timers
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // initialize the stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int *host_a, *host_b, *host_c;
    int *dev_a, *dev_b, *dev_c;

    // allocate the memory on the GPU
    cudaMalloc((void**)&dev_a, N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * sizeof(int));

    // allocate page-locked memory, used to stream
    cudaHostAlloc((void**)&host_a, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&host_b, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&host_c, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);

    for (int i = 0; i < FULL_DATA_SIZE; i++) {
        host_a[i] = rand();
        host_b[i] = rand();
    }

    // now loop over full data, in bite-sized chunks
    for (int i = 0; i < FULL_DATA_SIZE; i += N) {
        // copy the locked memory to the device, async
        cudaMemcpyAsync(dev_a, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dev_b, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream);
        
        kernel<<<N/256, 256, 0, stream>>>(dev_a, dev_b, dev_c);

        // copy the data from device to locked memory
        cudaMemcpyAsync(host_c + i, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost, stream);
    }

    cudaStreamSynchronize(stream);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Time taken: " << elapsedTime << std::endl;

    // cleanup the streams and memory
    cudaFreeHost(host_a);
    cudaFreeHost(host_b);
    cudaFreeHost(host_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    cudaStreamDestroy(stream);
    return 0;
}
