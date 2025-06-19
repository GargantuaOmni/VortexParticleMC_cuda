#include "monte_carlo_bs.hpp"
#include "device_vector_alias.hpp"
#include "vortex_particle_mc.hpp"
using namespace std;

inline float length(const float2& v)
{
    return std::sqrt(v.x * v.x + v.y * v.y);
}

inline float2 mis_contrib(const float2& dst,
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

MCBSEstimate monte_carlo_bs_cpu(
    const float2& dst, int sign, int auxiliary,
    const float2& dst_left,  const float2& dst_right,
    const float2& dst_top,   const float2& dst_bottom,
    const ParticleSet& P, const MCBSCtx& C, int N1, int N2,
    std::mt19937& rng)
{
    float w1 = float(N1) / float(N1 + N2);
    float w2 = float(N2) / float(N1 + N2);


    float2  s      {}, s_l {}, s_r {}, s_t {}, s_b {};
    float   cum    = (sign>0) ? C.cum_pos : C.cum_neg;
    if (C.periodic) cum *= 9.f;

    /* === part 1 : vorticity-PDF MIS branch === */
    for(int i=0;i<N1;++i){
        /* (1) 选粒子 idx via CDF */
        int idxLocal = sample_vorticity_idx_cdf(
                           rng,
                           (sign>0? C.cdf_pos->data():C.cdf_neg->data()),
                           (sign>0? C.cdf_pos->size():C.cdf_neg->size()));
        int pIdx = (sign>0? (*C.sub_pos)[idxLocal] : (*C.sub_neg)[idxLocal]);

        /* (2) 核内随机 r + Jacobian TODO: 先省略 → 直接用粒子中心 */
        float2 sp = P.pos[pIdx];
        if(C.periodic){
            float2 d = sample_periodic_deplace(rng);
            sp.x += d.x * C.Lx;
            sp.y += d.y * C.Ly;
        }

        float omega = evalVorticityAbs_cpu(
                         (sign>0? C.pos_view : C.neg_view),
                         sp, sign, false, false, C.h_w);

        float Rdst = SamplingDiskRadius(dst, C.periodic, C.Lx, C.Ly);
        float alpha = 1.f/(2.f*M_PI*Rdst);
        accumulate(s , mis_contrib(dst, sp, omega, w1,w2,cum, alpha));

        if(auxiliary){
            float alphaL = 1.f/(2.f*M_PI*SamplingDiskRadius(dst_left,  C.periodic,C.Lx,C.Ly));
            float alphaR = 1.f/(2.f*M_PI*SamplingDiskRadius(dst_right, C.periodic,C.Lx,C.Ly));
            float alphaT = 1.f/(2.f*M_PI*SamplingDiskRadius(dst_top,   C.periodic,C.Lx,C.Ly));
            float alphaB = 1.f/(2.f*M_PI*SamplingDiskRadius(dst_bottom,C.periodic,C.Lx,C.Ly));
            accumulate(s_l, mis_contrib(dst_left , sp, omega,w1,w2,cum,alphaL));
            accumulate(s_r, mis_contrib(dst_right, sp, omega,w1,w2,cum,alphaR));
            accumulate(s_t, mis_contrib(dst_top  , sp, omega,w1,w2,cum,alphaT));
            accumulate(s_b, mis_contrib(dst_bottom,sp, omega,w1,w2,cum,alphaB));
        }
    }

    /**********  Part-2 : 1/r-Disk 采样  **********/
    for(int i=0;i<N2;++i)
    {
        /* ---------- 主点 ---------- */
        float  R0   = SamplingDiskRadius(dst, C.periodic, C.Lx, C.Ly);
        float  a0   = 1.f/(2.f*M_PI*R0);
        float2 sp0  = sample_disk_biot_savart(rng, dst, R0);
        float  w0   = evalVorticityAbs_cpu((sign>0? C.pos_view:C.neg_view),
                                           sp0, sign,false,false,C.h_w);
        accumulate(s , mis_contrib(dst, sp0, w0, w1,w2,cum,a0));

        if(auxiliary){
            /* 左 */
            float RL = SamplingDiskRadius(dst_left, C.periodic,C.Lx,C.Ly);
            float aL = 1.f/(2.f*M_PI*RL);
            float2 spL = sample_disk_biot_savart(rng, dst_left, RL);
            float  wL  = evalVorticityAbs_cpu((sign>0? C.pos_view:C.neg_view),
                                              spL, sign,false,false,C.h_w);
            accumulate(s_l, mis_contrib(dst_left, spL, wL, w1,w2,cum,aL));

            /* 右 */
            float RR = SamplingDiskRadius(dst_right, C.periodic,C.Lx,C.Ly);
            float aR = 1.f/(2.f*M_PI*RR);
            float2 spR = sample_disk_biot_savart(rng, dst_right, RR);
            float  wR  = evalVorticityAbs_cpu((sign>0? C.pos_view:C.neg_view),
                                              spR, sign,false,false,C.h_w);
            accumulate(s_r, mis_contrib(dst_right, spR, wR, w1,w2,cum,aR));

            /* 上 */
            float RT = SamplingDiskRadius(dst_top, C.periodic,C.Lx,C.Ly);
            float aT = 1.f/(2.f*M_PI*RT);
            float2 spT = sample_disk_biot_savart(rng, dst_top, RT);
            float  wT  = evalVorticityAbs_cpu((sign>0? C.pos_view:C.neg_view),
                                              spT, sign,false,false,C.h_w);
            accumulate(s_t, mis_contrib(dst_top, spT, wT, w1,w2,cum,aT));

            /* 下 */
            float RB = SamplingDiskRadius(dst_bottom, C.periodic,C.Lx,C.Ly);
            float aB = 1.f/(2.f*M_PI*RB);
            float2 spB = sample_disk_biot_savart(rng, dst_bottom, RB);
            float  wB  = evalVorticityAbs_cpu((sign>0? C.pos_view:C.neg_view),
                                              spB, sign,false,false,C.h_w);
            accumulate(s_b, mis_contrib(dst_bottom, spB, wB, w1,w2,cum,aB));
        }
    }

    float invTot = 1.f / float(N1+N2);
    return {
        s     * invTot,
        s_l   * invTot,
        s_r   * invTot,
        s_t   * invTot,
        s_b   * invTot };
}
