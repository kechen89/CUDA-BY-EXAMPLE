#include<iostream>

#define imin(a,b) (a < b ? a : b)

const int N = 33 * 1024 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

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

float cuda_host_alloc_test(int size) {
    cudaEvent_t start, stop;
    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // allocate memory on the CPU side
    cudaHostAlloc((void**)&a, size*sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped);    
    cudaHostAlloc((void**)&b, size*sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped);
    cudaHostAlloc((void**)&partial_c, size*sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped);

    // fill in the host memory with data
    for (int i = 0; i < N; i++){
        a[i] = i;
        b[i] = i * 2;
    }

    cudaHostGetDevicePointer(&dev_a, a, 0);
    cudaHostGetDevicePointer(&dev_b, b, 0);
    cudaHostGetDevicePointer(&dev_partial_c, partial_c, 0);
    
    cudaEventRecord(start, 0);

    // launch kernel
    dot<<<blocksPerGrid, threadsPerBlock>>>(size, dev_a, dev_b, dev_partial_c);

    cudaThreadSynchronize(); 

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    //finish up on the CPU side
    c = 0;
    for (int i = 0; i < blocksPerGrid;i++){
        c += partial_c[i];
    }

    #define sum_squares(x) (x*(x+1)*(2*x+1)/6)
    std::cout << "Does GPU value " << c << " = " << 2 * sum_squares((float)(N - 1)) << " ? " << std::endl;

    // free memory on the CPU side
    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(partial_c);

    // free events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsedTime;
}

int main(void) {
    cudaDeviceProp prop;
    int whichDevice;
    cudaGetDevice(&whichDevice);
    cudaGetDeviceProperties(&prop, whichDevice);
    if (prop.canMapHostMemory != 1) {
        std::cout << "Device cannot map memory." << std::endl;
        return 0;
    }

    cudaSetDeviceFlags(cudaDeviceMapHost);

    float elapsedTime = cuda_host_alloc_test(N);
    std::cout << "Time using cudaHostAlloc: " << elapsedTime << std::endl;
}
