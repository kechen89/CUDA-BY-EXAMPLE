#include<iostream>

int main(void) {
    cudaDeviceProp prop;
    int dev;

    cudaGetDevice( &dev );
    std::cout << "ID of current CUDA device: " << dev << std::endl;

    memset( &prop, 0, sizeof( cudaDeviceProp ) );
    prop.major = 1;
    prop.minor = 3;
    cudaChooseDevice( &dev, &prop );
    std::cout << "ID of CUDA device closest to revision 1.3: " << dev << std::endl;
    cudaSetDevice( dev );

    return 0;
}
