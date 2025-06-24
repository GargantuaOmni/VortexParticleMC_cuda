#pragma once
#include <curand_kernel.h>
#include "vortex_particle_mc.hpp"
#include "monte_carlo_bs.hpp"
#include "device_vector_alias.hpp"
#include <curand_kernel.h>
#include "monte_carlo_bs.cuh"
#include <thrust/execution_policy.h>
#include <thrust/binary_search.h>

__device__ float evalVorticityAbs_gpu(const VorticityView& v,
                                      float2 p, int sign,
                                      bool use_signed, bool exclude_center,
                                      float h_w);

struct MCBSCtx_d {
    const int*   sub_pos;
    const int*   sub_neg;
    const float* cdf_pos;
    const float* cdf_neg;
    int          cnt_pos;
    int          cnt_neg;

    const float* icdf_kernel;
    int          kernel_L;
    float        h_w;

    float        cum_pos;
    float        cum_neg;

    float        Lx, Ly;
    bool         periodic;

    VorticityView pos_view;
    VorticityView neg_view;
};


__global__ void monte_carlo_bs_kernel(
        ParticleSet        ps,
        MCBSCtx_d          ctx,
        int                N1,
        int                N2,
        int                sign,            // +1 or -1 (NEW)
        float2*            out);



__global__ void monte_carlo_bs_kernel_safe(
        ParticleSet        ps,
        MCBSCtx_d          ctx,
        int                N1,
        int                N2,
        int                sign,            // +1 or -1 (NEW)
        float2*            out);

__device__ inline float length(float2 v){ return sqrtf(v.x*v.x + v.y*v.y); }

__device__ inline float sample_kernel_radius(curandStatePhilox4_32_10_t& rng,
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

__device__ inline int sample_vorticity_idx_cdf(curandStatePhilox4_32_10_t& rng,
                                        const float* cdf, int N)
{
    float u = curand_uniform(&rng);
    if(u >= 1.0f) u = 0.99999994f;
    
    int idx = thrust::upper_bound(thrust::seq, cdf, cdf+N, u) - cdf;

    if(idx == N) idx = N-1;
    return idx;
}

__device__ inline float2 sample_disk_biot_savart(curandStatePhilox4_32_10_t& rng,
                                          float2 center, float R)
{
    float phi = 2 * M_PI * curand_uniform(&rng);
    float r   = R * sqrtf(curand_uniform(&rng));    // CDF ∝ r²
    return {center.x + r*cosf(phi), center.y + r*sinf(phi)};
}
