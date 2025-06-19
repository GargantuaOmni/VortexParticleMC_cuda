#include "monte_carlo_bs.cuh"
#include "analytic_flow.hpp"
#include "utilities.hpp"
#include <cooperative_groups.h>

using namespace cooperative_groups;
#define TPB 128

__device__ float evalVorticityAbs_gpu(const VorticityView& v, float2 p,
                                      int sign, bool, bool, float h_w)
{
    float sum = 0.f;
    for(int i=0;i<v.N;++i){
        float2 q = v.pos[i];
        float  w = fabsf(v.w[i]);
        float2 r = {p.x-q.x, p.y-q.y};
        float  d = sqrtf(r.x*r.x + r.y*r.y);
        if(d < 2.f*h_w)
            sum += w * cubicSplinePDF(d/h_w);   //
    }
    return sum;
}

__device__ inline float2 mis_contrib_dev(const float2& dst,
                          const float2& sp,
                          float  omega,              // |ω(sp)|
                          float  w1, float  w2,      // N1 / (N1+N2),   N2 / (N1+N2)
                          float  cum,                // Σ|ω|
                          float  alpha)              // 1 / (2πR)
{
    if(omega <= 0.f) return {0.f, 0.f};

    float2 r = {dst.x - sp.x, dst.y - sp.y};
    float   r_norm = std::max(length(r), 1e-6f);
    float2  k = biot_savart_kernel(dst, sp);

    float denom = w1 * omega / cum + w2 * alpha / r_norm;   // Balanced Heuristic
    return k * (omega / denom);                    // omega * k / denom
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
        int                N1, int N2,
        int                sign,            // +1 or -1 (NEW)
        float2*            out)
{
    int id = blockIdx.x;
    if(id >= ps.N_cur) return;

    /* --- RNG per thread --- */
    curandStatePhilox4_32_10_t rng;
    curand_init(0x1234u + id, threadIdx.x, 0, &rng);

    __shared__ float2 ssum;
    if(threadIdx.x==0) ssum = {0,0};
    cooperative_groups::this_thread_block().sync();

    float2 dst       = ps.d_p()[id];
    float  Rmain     = ctx.Lx>ctx.Ly ? ctx.Lx : ctx.Ly;          // TODO:: Recheck
    float  alphamain = 1.f/(2.f*M_PI*Rmain);
    float  cum       = (sign>0 ? ctx.cum_pos : ctx.cum_neg);
    if(ctx.periodic) cum *= 9.f;
    float  w1 = float(N1)/(N1+N2);
    float  w2 = float(N2)/(N1+N2);

    /* ===== Part-1 : vorticity-PDF ===== */
    for(int t=threadIdx.x; t<N1; t+=blockDim.x)
    {
        int local = sample_vorticity_idx_cdf(rng,
                      sign>0 ? ctx.cdf_pos : ctx.cdf_neg,
                      sign>0 ? ctx.cnt_pos : ctx.cnt_neg);
        int gidx  = (sign>0? ctx.sub_pos[local]
                            : ctx.sub_neg[local]);

        float2 sp = ps.d_p()[gidx];

        if(ctx.periodic){
            float2 shift = sample_periodic_deplace(rng);
            sp.x += shift.x * ctx.Lx;
            sp.y += shift.y * ctx.Ly;
        }

        float omega = evalVorticityAbs_gpu(
                         sign>0? ctx.pos_view : ctx.neg_view,
                         sp, sign,false,false,ctx.h_w);

        float2 inc = mis_contrib_dev(dst, sp, omega,
                                     w1,w2,cum, alphamain);
        atomicAdd(&ssum.x, inc.x);
        atomicAdd(&ssum.y, inc.y);
    }

    /* ===== Part-2 : 1/r Disk ===== */
    for(int t=threadIdx.x; t<N2; t+=blockDim.x)
    {
        float2 sp = sample_disk_biot_savart(rng, dst, Rmain);
        float  omega = evalVorticityAbs_gpu(
                         sign>0? ctx.pos_view : ctx.neg_view,
                         sp, sign,false,false,ctx.h_w);

        float2 inc = mis_contrib_dev(dst, sp, omega,
                                     w1,w2,cum, alphamain);
        atomicAdd(&ssum.x, inc.x);
        atomicAdd(&ssum.y, inc.y);
    }

    cooperative_groups::this_thread_block().sync();
    if(threadIdx.x==0){
        out[id] = make_float2(ssum.x/(N1+N2), ssum.y/(N1+N2));
    }
}
