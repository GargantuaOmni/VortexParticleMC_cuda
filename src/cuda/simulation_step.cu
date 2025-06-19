#include "monte_carlo_bs.cuh"
#include <thrust/copy.h>

#define TPB 128

void Simulation::step_cuda()
{
    MCBSCtx_d dctx;
    dctx.sub_pos   = raw_ptr(sub_pos_);
    dctx.sub_neg   = raw_ptr(sub_neg_);
    dctx.cdf_pos   = raw_ptr(cdf_pos_);
    dctx.cdf_neg   = raw_ptr(cdf_neg_);
    dctx.cnt_pos   = sub_pos_.size();
    dctx.cnt_neg   = sub_neg_.size();
    dctx.icdf_kernel= raw_ptr(icdf_kernel);
    dctx.kernel_L  = icdf_kernel.size();
    dctx.h_w       = P_.h_w;
    dctx.cum_pos   = cum_pos_;
    dctx.cum_neg   = cum_neg_;
    dctx.Lx = P_.Lx; dctx.Ly = P_.Ly;
    dctx.periodic  = false;
    dctx.pos_view  = makePosView();
    dctx.neg_view  = makeNegView();

    dvec<float2> d_vel_p(ps.N_cur), d_vel_n(ps.N_cur);

    int blocks = ps.N_cur;
    monte_carlo_bs_kernel<<<blocks,TPB>>>(ps, dctx,
        P_.N1, P_.N2, /*sign=*/+1, raw_ptr(d_vel_p));
    monte_carlo_bs_kernel<<<blocks,TPB>>>(ps, dctx,
        P_.N1, P_.N2, /*sign=*/-1, raw_ptr(d_vel_n));
    cudaDeviceSynchronize();

    particles_.vel.resize(ps.N_cur);
    thrust::transform(d_vel_p.begin(), d_vel_p.end(),
                      d_vel_n.begin(),
                      particles_.vel.begin(),
                      [] __device__ (float2 a, float2 b){
                          return make_float2(a.x-b.x, a.y-b.y);
                      });

    thrust::transform(ps.pos.begin(), ps.pos.end(),
                      particles_.vel.begin(),
                      ps.pos.begin(),
                      [dt=P_.dt] __device__ (float2 p, float2 v){
                          return make_float2(p.x+v.x*dt, p.y+v.y*dt);
                      });

    thrust::for_each(ps.pos.begin(), ps.pos.end(),
                     [Lx=P_.Lx, Ly=P_.Ly] __device__ (float2& p){
                        p.x -= floorf(p.x / Lx) * Lx;
                        p.y -= floorf(p.y / Ly) * Ly;
                     });
}
