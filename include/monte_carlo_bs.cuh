#pragma once
#include <curand_kernel.h>
#include <math.h>
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
    float        InvLx;
    float        InvLy;
    bool         periodic;

    float        dx;

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
        int                auxiliary,  // 0:
        float2*            outC,       //
        float2*            outL,       //
        float2*            outR,       //
        float2*            outT,       //
        float2*            outB);


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
    float rhat   = icdf[i0]*(1.f-t) + icdf[i1]*t;
    return rhat * h;
}

__device__ inline float2 sample_kernel_vector(curandStatePhilox4_32_10_t& rng,
                                              const float* icdf, int L, float h)
{
    float r  = sample_kernel_radius(rng, icdf, L, h);
    float th = 2.f * M_PI * curand_uniform(&rng);
    float s, c; __sincosf(th, &s, &c);
    return make_float2(r * c, r * s);
}

__device__ inline float2 mul_mat2(float4 F, float2 v)
{
    return make_float2(F.x * v.x + F.y * v.y,
                       F.z * v.x + F.w * v.y);
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

/*
HD inline float wrap_delta(float d, float L, float InvL, bool periodic)
{
    if (!periodic) return d;
    float k = floorf(d * InvL + 0.5f);
    return d - k * L;
}
*/

HD inline float wrap_delta(float d, float L, float InvL, bool periodic)
{
    if (!periodic) return d;
    return d - roundf(d / L) * L;
}

__device__ inline float4 renorm_det1(float4 F){
    float det = F.x*F.w - F.y*F.z;
    float s = rsqrtf(fmaxf(det, 1e-12f));   // 不可压：det(F) → 1
    F.x*=s; F.y*=s; F.z*=s; F.w*=s;  return F;
}

__global__ void update_jacobian_from5(
        int N, float dt, float dx,
        const float2* velC, const float2* velL, const float2* velR,
        const float2* velT, const float2* velB,
        float4* Fout);


HD inline float wrap_min_image(float d, float L, bool periodic){
    if (!periodic) return d;
    return d - rintf(d / L) * L;
}

HD inline float2 bs_kernel_2d(float2 d, float eps2){
    // Kε(r) = (1/(2π)) * perp(d) / (|d|^2 + eps^2)
    float denom = d.x*d.x + d.y*d.y + eps2;
    float coef  = 0.5f * (1.0f / (float)M_PI) / denom;   // = 1/(2π*denom)
    return make_float2(-d.y * coef, d.x * coef);
}

__global__ void biot_savart_naive_kernel(
    const float2* __restrict__ pos,
    const float*  __restrict__ gamma,  //
    int N, float2* __restrict__ out,
    float Lx, float Ly, bool periodic,
    float eps);