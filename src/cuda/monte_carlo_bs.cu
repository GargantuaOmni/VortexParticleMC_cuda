#include "monte_carlo_bs.cuh"
#include "analytical_flow.hpp"
#include "utilities.hpp"
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cstdio>

namespace cg = cooperative_groups;

#define TPB 128

__device__ float evalVorticityAbs_gpu(const VorticityView& v, float2 p,
                                      int sign, bool, bool, float h_w, bool periodic)
{
    float sum = 0.f;
    if (periodic) {
        p.x -= floorf(p.x / v.Lx) * v.Lx;
        p.y -= floorf(p.y / v.Ly) * v.Ly;
    }
    for(int i=0;i<v.N;++i){
        float2 q = v.pos[i];
        float  w = fabsf(v.w[i]);
        float2 r = {wrap_delta(p.x-q.x, v.Lx, v.InvLx, periodic) , wrap_delta(p.y-q.y, v.Ly, v.InvLy, periodic)};
        float  d = sqrtf(r.x*r.x + r.y*r.y);
        if(d < 2.f*h_w)
            sum += w * cubicSplinePDF(d, h_w);   //
    }
    return sum;
}

__device__ inline float2 mis_contrib_dev(MCBSCtx_d ctx,
                          const float2& dst,
                          const float2& sp,
                          float  omega,              // |ω(sp)|
                          float  w1, float  w2,      // N1 / (N1+N2),   N2 / (N1+N2)
                          float  cum,                // Σ|ω|
                          float  alpha)              // 1 / (2πR)
{
    if(omega <= 0.f) return {0.f, 0.f};

    float2 r = { wrap_delta(dst.x - sp.x, ctx.Lx, ctx.InvLx, ctx.periodic),wrap_delta(dst.y - sp.y, ctx.Ly, ctx.InvLy, ctx.periodic) };
    float   r_norm = std::max(length(r), 1e-6f);
    float2  k = biot_savart_kernel(dst, sp);

    float denom = w1 * omega / cum + w2 * alpha / r_norm;   // Balanced Heuristic
    return (omega / denom) * k;                    // omega * k / denom
}

__device__ float2 sample_periodic_deplace(curandStatePhilox4_32_10_t& rng)
{
    float2 d{0,0};
    float ux = curand_uniform(&rng);
    float uy = curand_uniform(&rng);
    if(ux < 1.f/3.f)      d.x = -1.f;
    else if(ux > 2.f/3.f) d.x =  1.f;
    if(uy < 1.f/3.f)      d.y = -1.f;
    else if(uy > 2.f/3.f) d.y =  1.f;
    return d;
}
__global__ void monte_carlo_bs_kernel(
        ParticleSet        ps,
        MCBSCtx_d          ctx,
        int                N1,
        int                N2,
        int                sign,          // +1 / -1
        float2*            out)
{
    int gid = blockIdx.x;                 // 1
    if(gid >= ps.N_cur) return;
    cg::thread_block tb = cg::this_thread_block();   // <--

    curandStatePhilox4_32_10_t rng;
    curand_init(0x1234u + gid,          // seed
                threadIdx.x,            // subsequence = lane id
                0, &rng);

    __shared__ float2 ssum;
    if(tb.thread_rank()==0) ssum = make_float2(0.f, 0.f);
    tb.sync();                            //

    float2 dst  = ps.d_p()[gid];
    float  Rdst = SamplingDiskRadius(dst, ctx.periodic, ctx.Lx, ctx.Ly);
    float  alpha_dst = 1.f / (2.f*M_PI*Rdst);

    float  cum  = sign>0 ? ctx.cum_pos : ctx.cum_neg;
    if(ctx.periodic) cum *= 9.f;

    float  w1   = float(N1) / float(N1+N2);
    float  w2   = float(N2) / float(N1+N2);

    /* ======================================================           */
    for(int t = threadIdx.x; t < N1; t += blockDim.x)
    {
        int idLocal = sample_vorticity_idx_cdf(
                         rng,
                         sign>0 ? ctx.cdf_pos : ctx.cdf_neg,
                         sign>0 ? ctx.cnt_pos : ctx.cnt_neg);
        int pIdx    = (sign>0 ? ctx.sub_pos[idLocal]
                              : ctx.sub_neg[idLocal]);

        float2 sp = ps.d_p()[pIdx];
        if(ctx.periodic){
            float2 sh = sample_periodic_deplace(rng);
            sp.x += sh.x * ctx.Lx;
            sp.y += sh.y * ctx.Ly;
        }

        float omega = evalVorticityAbs_gpu(
                        sign>0 ? ctx.pos_view : ctx.neg_view,
                        sp, sign, false, false, ctx.h_w, ctx.periodic);

        float2 inc = mis_contrib_dev(ctx, dst, sp, omega,
                                     w1, w2, cum, alpha_dst);
        atomicAdd(&ssum.x, inc.x);
        atomicAdd(&ssum.y, inc.y);
    }
    tb.sync();

    /* ======================================================     */
    for(int t = threadIdx.x; t < N2; t += blockDim.x)
    {
        float2 sp  = sample_disk_biot_savart(rng, dst, Rdst);
        float  omega = evalVorticityAbs_gpu(
                         sign>0 ? ctx.pos_view : ctx.neg_view,
                         sp, sign, false, false, ctx.h_w, ctx.periodic);

        float2 inc = mis_contrib_dev(ctx, dst, sp, omega,
                                     w1, w2, cum, alpha_dst);
        atomicAdd(&ssum.x, inc.x);
        atomicAdd(&ssum.y, inc.y);
    }
    tb.sync();

    if(tb.thread_rank()==0)
        out[gid] = make_float2(ssum.x / float(N1+N2),
                               ssum.y / float(N1+N2));
}


__global__ void monte_carlo_bs_kernel_safe(
        ParticleSet        ps,
        MCBSCtx_d          ctx,
        int                N1,
        int                N2,
        int                sign,
        float2*            out)
{

    int gid = blockIdx.x;
    if(gid >= ps.N_cur) return;

    curandStatePhilox4_32_10_t rng;
    curand_init(0x1234u + gid, threadIdx.x, 0, &rng);

    float2 dst  = ps.d_p()[gid];
    float  Rdst = SamplingDiskRadius(dst, ctx.periodic, ctx.Lx, ctx.Ly);
    float  alpha_dst = 1.f / (2.f * M_PI * Rdst);

    float  cum  = (sign > 0 ? ctx.cum_pos : ctx.cum_neg);
    if(ctx.periodic) cum *= 9.f;


    float  w1   = float(N1) / float(N1 + N2);
    float  w2   = float(N2) / float(N1 + N2);

    float2 local_acc = make_float2(0.f, 0.f);

    /* ---------- Part-1 : vorticity-PDF ---------- */
    for (int t = threadIdx.x; t < N1; t += blockDim.x)
    {
        int local = sample_vorticity_idx_cdf(
                        rng,
                        sign > 0 ? ctx.cdf_pos : ctx.cdf_neg,
                        sign > 0 ? ctx.cnt_pos : ctx.cnt_neg);
        int pIdx  = (sign > 0 ? ctx.sub_pos[local] : ctx.sub_neg[local]);

        float2 sp = ps.d_p()[pIdx];
        if(ctx.periodic){
            float2 sh = sample_periodic_deplace(rng);
            sp.x += sh.x * ctx.Lx;
            sp.y += sh.y * ctx.Ly;
        }

        float omega = evalVorticityAbs_gpu(
                        sign > 0 ? ctx.pos_view : ctx.neg_view,
                        sp, sign, false, false, ctx.h_w, ctx.periodic);

        float2 inc = mis_contrib_dev(ctx, dst, sp, omega, w1, w2, cum, alpha_dst);
        local_acc.x += inc.x;
        local_acc.y += inc.y;

    }
    /* ---------- Part-2 : 1/r-Disk ---------- */
    for (int t = threadIdx.x; t < N2; t += blockDim.x)
    {
        float2 sp  = sample_disk_biot_savart(rng, dst, Rdst);


        float omega = evalVorticityAbs_gpu(
                        sign > 0 ? ctx.pos_view : ctx.neg_view,
                        sp, sign, false, false, ctx.h_w, ctx.periodic);

        float2 inc = mis_contrib_dev(ctx, dst, sp, omega, w1, w2, cum, alpha_dst);
        local_acc.x += inc.x;
        local_acc.y += inc.y;

    }

    float norm = 1.f / float(N1 + N2);
    atomicAdd(&out[gid].x, local_acc.x * norm);
    atomicAdd(&out[gid].y, local_acc.y * norm);
}