// include/device_vector_alias.hpp
#pragma once
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cpp/memory.h>   // CPU fallback


// device_vector_alias.hpp
#pragma once
#include <vector>
#if defined(USE_CUDA) && defined(__CUDACC__)
  #include <cuda_runtime.h>         // float2 / float3 / …
  template<typename T> using dvec = thrust::device_vector<T>;
#else
#ifndef __VECTOR_TYPES_H__                // CUDA 自带头的 include-guard
struct float2 { float x, y; };
#endif
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


template<typename Vec>
auto raw_ptr(const Vec& v)                   //
{
#if defined(USE_CUDA) && defined(__CUDACC__)
    return thrust::raw_pointer_cast(v.data());      // const T*
#else
    return v.data();                                // const T*
#endif
}