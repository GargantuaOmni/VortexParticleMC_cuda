#include <curand_kernel.h>
#include "monte_carlo_bs.cuh"
#include "device_vector_alias.hpp"
#include <thrust/execution_policy.h>

__device__ inline float length(float2 v){ return sqrtf(v.x*v.x + v.y*v.y); }

__device__ float sample_kernel_radius(curandStatePhilox4_32_10_t& rng,
                                      const float* icdf, int L, float h)
{
    float u   = curand_uniform(&rng);
    float f   = u * (L - 1);
    int   i0  = (int)f;
    int   i1  = min(i0 + 1, L - 1);
    float t   = f - i0;
    float r̂   = icdf[i0]*(1.f-t) + icdf[i1]*t;
    return r̂ * h;
}

__device__ int sample_vorticity_idx_cdf(curandStatePhilox4_32_10_t& rng,
                                        const float* cdf, int N)
{
    float u = curand_uniform(&rng);
    // device 端 thrust::upper_bound
    return thrust::upper_bound(thrust::seq, cdf, cdf+N, u) - cdf;
}

__device__ float2 sample_disk_biot_savart(curandStatePhilox4_32_10_t& rng,
                                          float2 center, float R)
{
    float phi = TWO_PI * curand_uniform(&rng);
    float r   = R * sqrtf(curand_uniform(&rng));    // CDF ∝ r²
    return {center.x + r*cosf(phi), center.y + r*sinf(phi)};
}
