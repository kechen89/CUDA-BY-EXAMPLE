#include <iostream>
#include "cuda.h"
#include "../common/cpu_bitmap.h"

#define INF 2e10f

struct Sphere{
    float r,b,g;
    float radius;
    float x,y,z;
    __device__ float hit(float ox, float oy, float *n){
        float dx = ox - x;
        float dy = oy - y;
        if (dx * dx + dy * dy  < radius * radius){
            float dz = sqrtf(radius * radius - dx * dx - dy * dy);
            *n = dz / sqrtf(radius * radius);
            return dz + z;
        }
        return -INF;
    }
};

#define rnd(x) (x * rand() / RAND_MAX)
#define SPHERES 20

__constant__ Sphere s[SPHERES];

__global__ void kernel(unsigned char *ptr);

int main(void){
    // capture the start time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    CPUBitmap bitmap(DIM, DIM);
    unsigned char *dev_bitmap;

    // allocate memory on the GPU for the output bitmap
    cudaMalloc((void**)&dev_bitmap, bitmap.image_size());

    // allocate temp memory, initialize it, copy to memory on the GPU, and then free our temp memory
    Sphere *temp_s = (Sphere*)malloc(sizeof(Sphere) * SPHERES);
    for (int i = 0; i < SPHERES; i++) {
        temp_s[i].r = rnd(1.0f);
        temp_s[i].g = rnd(1.0f);
        temp_s[i].b = rnd(1.0f);
        temp_s[i].x = rnd(1000.0f) - 500;
        temp_s[i].y = rnd(1000.0f) - 500;
        temp_s[i].z = rnd(1000.0f) - 500;
        temp_s[i].radius = rnd(100.0f) + 20;
    }
    cudaMemcpyToSymbol(s, temp_s, sizeof(Sphere) * SPHERES);

    free(temp_s);

    // generate a bitmap from our sphere data
    dim3 grids(DIM/16, DIM/16);
    dim3 threads(16,16);
    kernel<<<grids,threads>>>(s, dev_bitmap);

    // copy our bitmap back from the GPU for display
    cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);

    // get stop time, and display the timing results
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapseTime(&elapsedTime, start, stop);
   
    std::cout << "Time to generate: " << elapsedTime << "ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // display
    bitmap.display_and_exit();

    // free our memory
    cudaFree(dev_bitmap);
    cudaFree(s);
}

__global__ void kernel(unsigned char *ptr){
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    float ox = (x - DIM/2);
    float oy = (y - DIM/2);

    float r = 0, g = 0, b = 0;
    float maxz = -INF;

    for (int i = 0; i < SPHERES; i++){
        float n;
        float t = s[i].hit(ox, oy, &n);
        if (t > maxz){
            float fscale = n;
            r = s[i].r * fscale;
            g = s[i].g * fscale;
            b = s[i].b * fscale;
        }
    }    

    ptr[offset*4 + 0] = (int)(r * 255);
    ptr[offset*4 + 1] = (int)(g * 255);
    ptr[offset*4 + 2] = (int)(b * 255);
    ptr[offset*4 + 3] = 255;
}
