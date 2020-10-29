#include "cuda.h"
#include "../common/cpu_anim.h"
#include <iostream>

#define DIM 1024
#define PI 3.1415926535897932f
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f
#define SPEED 0.25f

// these exist on the GPU side
texture<float,2> texConstSrc;
texture<float,2> texIn;
texture<float,2> texOut;


// globals needed by the update routine
struct DataBlock{
    unsigned char *output_bitmap;
    float *dev_inSrc;
    float *dev_outSrc;
    float *dev_constSrc;
    CPUAnimBitmap *bitmap;
    cudaEvent_t start, stop;
    float totalTime;
    float frames;
}

__global__ void copy_const_kernel(float *iptr, const float *cptr){
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float c = tex2D(texConstSrc, x, y);
    if (c != 0) 
        iptr[offset] = c;
}

__global__ void blend_kernel(float *dst, const float *dstOut){
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float t, l, c, r, b;
   
    if (dstOut){
        t = tex2D(texIn, x, y - 1);
        l = tex2D(texIn, x - 1, y);
        c = tex2D(texIn, x, y);
        r = tex2D(texIn, x + 1, y);
        b = tex2D(texIn, x, y + 1);
    } else {
        t = tex2D(texOut, x, y - 1);
        l = tex2D(texOut, x - 1, y);
        c = tex2D(texOut, x, y);
        r = tex2D(texOut, x + 1, y);
        b = tex2D(texOut, x, y + 1);
    }
    dst[offset] = c + SPEED * (t + b + r + l - 4 * c);
}

void anim_gpu(DataBlock *d, int ticks){
    cudaEventRecord(d->start, 0);
    dim3 blocks(DIM/16, DIM/16);
    dim3 threads(16, 16);
    CPUAnimBitmap *bitmap = d->bitmap;

    // since tex is global and bound, we have to use a flag to select which is in/out per iteration
    volatile bool dstOut = true;
    for (int i = 0; i < 90; i++){
        float *in, *out;
        if (dstOut){
            in = d->dev_inSrc;
            out = d->dev_outSrc;
        }else{
            out = d->dev_inSrc;
            in = d->dev_outSrc;
        }
        copy_const_kernel<<<blocks,threads>>>(in);
        blend_kernel<<<blocks,threads>>>(out,dstOut);
        dstOut = !dstOut;
    }
   
    float_to_color<<<blocks,threads>>>(d->output_bitmap,d->dev_inSrc);

    cudaMemcpy(bitmap->get_ptr(),d->output_bitmap,bitmap->image_size(),cudaMemcpyDeviceToHost);
    cudaEventRecord(d->stop,0);
    cudaEventSynchronize(d->stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, d->start, d->stop);

    d->totalTime += elapsedTime;
    ++d -> frames;
    std::cout << "Average Time per frame: " << d->totalTime/d->frames << "ms" << std::endl;
}

void anim_exit(DataBlock *d){
    cudaUnbindTexture(texIn);
    cudaUnbindTexture(texOut);
    cudaUnbindTexture(texConstSrc);

    cudaFree(d->dev_inSrc);
    cudaFree(d->dev_outSrc);
    cudaFree(d->dev_constSrc);

    cudaEventDestroy(d->start);
    cudaEventDestroy(d->stop);
}

int main(void){
    DataBlock data;
    CPUAnimBitmap bitmap(DIM, DIM, &data);
    data.bitmap = &bitmap;
    data.totalTime = 0;
    data.frames = 0;
    cudaEventCreate(&data.start);
    cudaEventCreate(&data.stop);

    cudaMalloc((void**)&data.output_bitmap, bitmap.image_size());

    // assume float == 4 chars in size (i.e., rgba)
    cudaMalloc((void**)&data.dev_inSrc, bitmap.image_size());
    cudaMalloc((void**)&data.dev_outSrc, bitmap.image_size());
    cudaMalloc((void**)&data.dev_constSrc, bitmap.image_size());

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    cudaBindTexture2D(NULL, texConstSrc, data.dev_constSrc, desc, DIM, DIM, sizeof(float) * DIM);
    cudaBindTexture2D(NULL, texIn, data.dev_inSrc, desc, DIM, DIM, sizeof(float) * DIM);
    cudaBindTexture2D(NULL, texOut, data.dev_outSrc, desc, DIM, DIM, sizeof(float) * DIM);    

    // initialize the constant data
    float *temp = (float*)malloc(bitmap.image_size());
    for (int i = 0; i < DIM * DIM; i++){
        temp[i] = 0;
        int x = i % DIM;
        int y = i / DIM;
        if ((x > 300) && (x < 600) && (y > 310) && (y < 601))
            temp[i] = MAX_TEMP;
    }
    
    temp[DIM * 100 + 100] = (MAX_TEMP + MIN_TEMP)/2;
    temp[DIM * 700 + 100] = MIN_TEMP;
    temp[DIM * 300 + 300] = MIN_TEMP;
    temp[DIM * 200 + 700] = MIN_TEMP;

    for (int y = 800; y < 900; y++)
        for (int x = 400; x < 500; x++)
            temp[x + y * DIM] = MIN_TEMP;

    cudaMemcpy(data.dev_constSrc,temp,bitmap.image_size(),cudaMemcpyHostToDevice);

    for (int y = 800; y < DIM; y++)
        for (int x = 0; x < 200; x++)
            temp[x + y * DIM] = MAX_TEMP;

    cudaMemcpy(data.dev_inSrc, temp, bitmap.image_size(),cudaMemcpyHostToDevice);

    free(temp);

    bitmap.anim_and_exit( (void(*)(void*, int))anim_gpu, (void(*)(void*))anim_exit);
}
