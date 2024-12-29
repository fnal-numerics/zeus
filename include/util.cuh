#ifndef gpuprint_util_h
#define gpuprint_util_h

#include <stdio.h>
#include <iostream>
#include <stdarg.h>
#include <string>
#include <array>

template <class T, size_t numArrays>
void
display(size_t arraySize, ...)
{
    va_list params;
    va_start(params, arraySize);
    
    std::array<T*, numArrays> hArrays;
    
    for(auto &array:hArrays){
        array = (T*)malloc(sizeof(T) * arraySize);
        cudaMemcpy(array, va_arg(params, T*), sizeof(T) * arraySize, cudaMemcpyDeviceToHost);
    }
    
    va_end(params);
    
    auto PrintSameIndex = [hArrays](size_t index){
        for(auto &array:hArrays){
            std::cout<< array[index] << "\t";
        }   
        std::cout << std::endl;
    };
    
    for(size_t i=0; i<arraySize; ++i)
        PrintSameIndex(i); 
    
    for(auto &array:hArrays)
        free(array);
}
#endif
