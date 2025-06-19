// include/device_vector_alias.hpp
#pragma once
#include <vector>
#if defined(USE_CUDA)
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cpp/memory.h>   // CPU fallback
#endif

// device_vector_alias.hpp
#include <vector>
#if defined(USE_CUDA)
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
#if defined(USE_CUDA)
    return thrust::raw_pointer_cast(v.data());
#else
    return v.data();
#endif
}

template<typename Vec>
auto raw_ptr(const Vec& v)                   //
{
#if defined(USE_CUDA)
    return thrust::raw_pointer_cast(v.data());      // const T*
#else
    return v.data();                                // const T*
#endif
}

#ifdef USE_CUDA
#define HD __host__ __device__
#else
#define HD
#endif


#ifndef __device__
#define __device__
#endif


#ifndef __host__
#define __host__
#endif

/* Override operators for float2 */
inline float2 operator+(float2 a, float2 b){ return {a.x+b.x, a.y+b.y}; }
inline float2 operator-(float2 a, float2 b){ return {a.x-b.x, a.y-b.y}; }
inline float2& operator+=(float2& a, float2 b){ a.x+=b.x; a.y+=b.y; return a; }
inline float2  operator*(float2 a, float s)   { return {a.x*s, a.y*s}; }
inline float2  operator*(float a, float2 s)   { return {a*s.x, a*s.y}; }
inline float2& operator*=(float2& a, float s) { a.x*=s; a.y*=s; return a; }