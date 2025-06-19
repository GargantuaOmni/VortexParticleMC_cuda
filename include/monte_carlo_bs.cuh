#pragma once
#include <curand_kernel.h>
#include "vortex_particle_mc.hpp"
#include "monte_carlo_bs.hpp"

__device__ float evalVorticityAbs_gpu(const VorticityView& v,
                                      float2 p, int sign,
                                      bool use_signed, bool exclude_center,
                                      float h_w);

__global__ void monte_carlo_bs_kernel(
        const ParticleSet  ps,
        MCBSCtx            ctx,
        int                N1, int N2,
        float2*            out_vel);


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
