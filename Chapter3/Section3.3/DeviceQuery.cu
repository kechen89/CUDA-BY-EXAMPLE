#include<iostream>

int main(void) {
    cudaDeviceProp prop;   
    int count;

    cudaGetDeviceCount(&count);

    for (int i = 0; i < count; i++) {
        cudaGetDeviceProperties(&prop, i);
        std::cout << "General Information for device" << i << std::endl;
        std::cout << "Name:" << prop.name << std::endl;
        std::cout << "Compute capability:" << prop.major << prop.minor << std::endl;
    } 

}
