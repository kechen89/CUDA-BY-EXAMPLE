#include <iostream>

#define N (33 * 1024)

__global__ void add(int *a, int *b, int *c){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N){
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }
}

int main(void){
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;
    
    // allocate the memory on the GPU
    cudaMalloc((void**)&dev_a, N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * sizeof(int));

    // fill the arrays 'a' and 'b' on the CPU
    for (int i = 0; i < N; i++){
        a[i] = i;
        b[i] = i * i;
    }

    // copy the arrays 'a' and 'b' to the GPU
    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    // launch kernel
    add<<<128,128>>>(dev_a, dev_b, dev_c);

    // copy the array 'c' back from the GPU to the CPU
    cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // verify that the GPU did the work we requested
    bool success = true;
    for (int i = 0; i < N; i++){
        if ((a[i] + b[i]) != c[i]){
            std::cout << "Error: " << a[i] << " + " << b[i] << " != " << c[i] << std::endl;
            success = false;
        }
    }
    if (success) std::cout << "We did it!" << std::endl;

    // free the memory allocated on the GPU
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
