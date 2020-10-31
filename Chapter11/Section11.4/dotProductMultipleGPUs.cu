#include<iostream>
#include "../../common/book.h"

#define imin(a,b) (a < b ? a : b)

const int N = 33 * 1024 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

struct DataStruct {
    int deviceID;
    int size;
    float *a;
    float *b;
    float returnValue;
};

__global__ void dot(int size, float *a, float *b, float *c){
    __shared__ float cache[threadsPerBlock];
   int tid = threadIdx.x + blockIdx.x * blockDim.x;
   int cacheIndex = threadIdx.x;

   float temp = 0;
   while (tid < size){
       temp += a[tid] * b[tid];
       tid += blockDim.x * gridDim.x;
   }

   // set the cache values
   cache[cacheIndex] = temp;

   // synchronize threads in this block
   __syncthreads();

   // for reductions, threadsPerBlock must be a power of 2
   // because of the following code
   int i = blockDim.x/2;
   while (i != 0){
       if (cacheIndex < i)
           cache[cacheIndex] += cache[cacheIndex + i];
       __syncthreads();
       i /= 2;
   }

   if (cacheIndex == 0)
       c[blockIdx.x] = cache[0];
}

void* routine(void *pvoidData) {
    DataStruct *data = (DataStruct*)pvoidData;
    
    if (data->deviceID != 0) {
        cudaSetDevice(data->deviceID);
        cudaSetDeviceFlags(cudaDeviceMapHost);
    }

    int size = data->size;
    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;

    // allocate memory on the CPU side
    a = data->a;
    b = data->b;
    partial_c = (float*)malloc(blocksPerGrid * sizeof(float));

    // allocate the memory on the GPU
    cudaHostGetDevicePointer(&dev_a, a, 0);
    cudaHostGetDevicePointer(&dev_b, b, 0);
    cudaMalloc((void**)&dev_partial_c, blocksPerGrid * sizeof(float));

    // launch kernel
    dot<<<blocksPerGrid, threadsPerBlock>>>(size, dev_a, dev_b, dev_partial_c);

   // copy the array 'c' back from the GPU to the CPU
   cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid*sizeof(float),cudaMemcpyDeviceToHost);
 
   //finish up on the CPU side
   c = 0;
   for (int i = 0; i < blocksPerGrid;i++){
       c += partial_c[i];
   }

   // free memory on the GPU side
   cudaFree(dev_partial_c);

   // free memory on the CPU side
   free(partial_c);
   
   data->returnValue = c;
   return 0;
}

int main(void) {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount < 2) {
        std::cout << "We need at least two compute 1.0 or greater devices, but only found " << deviceCount << std::endl;
        return 0;
    }

    cudaDeviceProp prop;
    for (int i = 0; i < 2; i++){
        cudaGetDeviceProperties(&prop, i);
        if (prop.canMapHostMemory != 1) {
            std::cout << "Device " << i << " cannot map memory." << std::endl;
            return 0;
        }
    }    

    float *a, *b;
    cudaSetDevice(0);
    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaHostAlloc((void**)&a, N * sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocPortable | cudaHostAllocMapped);
    cudaHostAlloc((void**)&b, N * sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocPortable | cudaHostAllocMapped);    

    // fill in the host memory with data
    for (int i = 0; i < N; i++){
        a[i] = i;
        b[i] = i * 2;
    }

    DataStruct data[2];
    data[0].deviceID = 0;
    data[0].size = N / 2;
    data[0].a = a;
    data[0].b = b;

    data[1].deviceID = 1;
    data[1].size = N / 2;
    data[1].a = a + N / 2;
    data[1].b = b + N / 2;

    CUTThread thread = start_thread(routine, &(data[1]));
    routine(&(data[0]));

    end_thread(thread);    

    // free memory on the CPU side
    cudaFreeHost(a);
    cudaFreeHost(b);

    std::cout << "Value calculated: " << data[0].returnValue + data[1].returnValue << std::endl;

    return 0;
}
