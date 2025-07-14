#pragma once
#include <cmath>
#include <device_vector_alias.hpp>

constexpr float TWO_PI = 6.28318530717958647692f;

/* =========  Taylor–Green 2D =========
   u(x,y) =  sin(2πx) cos(2πy)
   v(x,y) = -cos(2πx) sin(2πy)
   ω(x,y) = ∂v/∂x - ∂u/∂y = 4π sin(2πx) sin(2πy)                                 */

__host__ __device__ inline float2 taylor_green_velocity(float x, float y)
{
    return {
        std::sin(TWO_PI * x) * std::cos(TWO_PI * y),           // u
       -std::cos(TWO_PI * x) * std::sin(TWO_PI * y)            // v
    };
}
__host__ __device__ inline float2 taylor_green_velocity(const float2& p){
    return taylor_green_velocity(p.x, p.y);
}

/*  ω = 4π sin(2πx) sin(2πy) */
__host__ __device__ inline float taylor_green_vorticity(float x, float y)
{
    return 4.0f * static_cast<float>(M_PI) *
           std::sin(TWO_PI * x) * std::sin(TWO_PI * y);
}
__host__ __device__ inline float taylor_green_vorticity(const float2& p){
    return taylor_green_vorticity(p.x, p.y);
}
