#pragma once
#include <random>
#include "device_vector_alias.hpp"   // dvec
#include "utilities.hpp"             // cubicSplinePDF, SamplingDiskRadius, etc.
#include "vortex_particle_mc.hpp"    // ParticleSet

struct MCBSEstimate {
    float2 main   {};
    float2 left   {};
    float2 right  {};
    float2 top    {};
    float2 bottom {};
};

struct MCBSCtx {
    /* === read-only buffers constructed elsewhere === */
    const dvec<int>*   sub_pos   = nullptr;   // Positive particles idx
    const dvec<int>*   sub_neg   = nullptr;   // Negative particles idx
    const dvec<float>* cdf_pos   = nullptr;   // normalized cdf
    const dvec<float>* cdf_neg   = nullptr;
    const dvec<float>* icdf_kernel = nullptr; // iCDF for the kernel function
    int                kernel_L  = 0;         // length of iCDF table
    float              h_w       = 1.0f;      // h_w
    bool               periodic  = false;
    float              Lx        = 1.0f;
    float              Ly        = 1.0f;
    float              InvLx        = 1.0f;
    float              InvLy        = 1.0f;
    /* === scalar pre-computed in build_subsets_cpu() === */
    float              cum_pos   = 1.0f;      // Σ|ω|  positive
    float              cum_neg   = 1.0f;      // Σ|ω|  negative

    VorticityView      pos_view;
    VorticityView      neg_view;
};

MCBSEstimate monte_carlo_bs_cpu(
    const float2& dst,            // destination
    int  sign,                    // +1 / -1
    int  auxiliary,               //
    const float2& dst_left,
    const float2& dst_right,
    const float2& dst_top,
    const float2& dst_bottom,
    const ParticleSet& P,
    const MCBSCtx& ctx,
    int N1, int N2,               // Number of sampleing
    std::mt19937& rng             // Random seed
);

float sample_kernel_radius(std::mt19937 &gen,
                                  const float *icdf, int L, float h);

int sample_vorticity_idx_cdf(std::mt19937& gen,
                                    const float* cdf, int N);

float2 sample_disk_biot_savart(std::mt19937& gen,
                                      const float2& center, float R);

__device__ inline float2 biot_savart_kernel(const float2 &x, const float2 &y)
{
    float2 r {x.x - y.x, x.y - y.y};
    float  inv_r2 = 1.f / (r.x * r.x + r.y * r.y + 1e-8f);
    return { -r.y * inv_r2 / (2*float(M_PI)),
              +r.x * inv_r2 / (2*float(M_PI)) };
}

static inline void accumulate(float2& acc, const float2& add)
{
    acc.x += add.x; acc.y += add.y;
}

__host__ __device__ inline float SamplingDiskRadius(const float2& dst,
                                bool periodic,
                                const float & Lx, const float & Ly)
{
    if(!periodic)                    //
        return std::sqrt(Lx*Lx + Ly*Ly);

    //
    float maxR2 = 0.f;
    for(int dx=-1; dx<=1; ++dx)
        for(int dy=-1; dy<=1; ++dy){
            float2 img { dst.x + dx*Lx, dst.y + dy*Ly };
            float2 d   { img.x - dst.x, img.y - dst.y };
            maxR2 = std::max(maxR2, d.x*d.x + d.y*d.y);
        }
    return std::sqrt(maxR2);
}


static float2 sample_periodic_deplace(std::mt19937& gen)
{
    std::uniform_real_distribution<float> U(0.f, 1.f);
    float2 d{0,0};
    float ux = U(gen), uy = U(gen);
    if(ux < 1.f/3.f)      d.x = -1.f;
    else if(ux > 2.f/3.f) d.x =  1.f;
    if(uy < 1.f/3.f)      d.y = -1.f;
    else if(uy > 2.f/3.f) d.y =  1.f;
    return d;                    // (−1,0,+1)²
}
