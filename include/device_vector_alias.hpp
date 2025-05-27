// include/device_vector_alias.hpp
#pragma once
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cpp/memory.h>   // CPU fallback


#if defined(USE_CUDA) && defined(__CUDACC__)
template<typename T> using dvec = thrust::device_vector<T>;
#else
template<typename T> using dvec = std::vector<T>;
#endif


template<typename Vec>
auto raw_ptr(Vec& v){
#if defined(USE_CUDA) && defined(__CUDACC__)
    return thrust::raw_pointer_cast(v.data());
#else
    return v.data();
#endif
}
