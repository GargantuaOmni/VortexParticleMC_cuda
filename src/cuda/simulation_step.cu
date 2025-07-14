#include "monte_carlo_bs.cuh"
#include "device_vector_alias.hpp"
#include "vortex_particle_mc.hpp"
#include <thrust/copy.h>
#include <iostream>

#define TPB 128


void Simulation::step_cuda(bool periodic)
{
    std::cout << "Simulation::step_cuda" << std::endl;
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
    dctx.InvLx = 1.0 / P_.Lx; dctx.InvLy = 1.0 / P_.Ly;
    dctx.periodic  = periodic;
    dctx.pos_view  = makePosView();
    dctx.neg_view  = makeNegView();

    std::cout << "debug 01" << std::endl;

#if defined(USE_CUDA)
    std::cout << "debug 01" << std::endl;
#endif

    dvec<float2> d_vel_p(particles_.N_cur), d_vel_n(particles_.N_cur);

    int blocks = particles_.N_cur;
    cudaMemset(raw_ptr(d_vel_p), 0, particles_.N_cur*sizeof(float2));
    cudaMemset(raw_ptr(d_vel_n), 0, particles_.N_cur*sizeof(float2));

    monte_carlo_bs_kernel_safe<<<blocks,TPB>>>(particles_, dctx,
        P_.N1, P_.N2, /*sign=*/+1, raw_ptr(d_vel_p));
    monte_carlo_bs_kernel_safe<<<blocks,TPB>>>(particles_, dctx,
        P_.N1, P_.N2, /*sign=*/-1, raw_ptr(d_vel_n));
    cudaDeviceSynchronize();

    std::cout << "debug 02" << std::endl;

    particles_.vel.resize(particles_.N_cur);
    thrust::transform(d_vel_p.begin(), d_vel_p.end(),
                      d_vel_n.begin(),
                      particles_.vel.begin(),
                      [] __device__ (float2 a, float2 b){
                          return make_float2(a.x-b.x, a.y-b.y);
                      });

    thrust::transform(particles_.pos.begin(), particles_.pos.end(),
                      particles_.vel.begin(),
                      particles_.pos.begin(),
                      [dt=P_.dt] __device__ (float2 p, float2 v){
                          return make_float2(p.x+v.x*dt, p.y+v.y*dt);
                      });

    thrust::for_each(particles_.pos.begin(), particles_.pos.end(),
                     [Lx=P_.Lx, Ly=P_.Ly] __device__ (float2& p){
                        p.x -= floorf(p.x / Lx) * Lx;
                        p.y -= floorf(p.y / Ly) * Ly;
                     });

    std::cout << "debug 03" << std::endl;
}
