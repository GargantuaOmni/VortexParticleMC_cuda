#include "monte_carlo_bs.cuh"
#include "device_vector_alias.hpp"
#include "vortex_particle_mc.hpp"
#include <thrust/copy.h>
#include <iostream>

#define TPB 128

void Simulation::step_cuda(bool periodic)
{
    const int N = particles_.N_cur;
    // const int TPB = 128;
    dim3 grid(N);                    //

    MCBSCtx_d dctx{};
    dctx.sub_pos   = raw_ptr(sub_pos_);
    dctx.sub_neg   = raw_ptr(sub_neg_);
    dctx.cdf_pos   = raw_ptr(cdf_pos_);
    dctx.cdf_neg   = raw_ptr(cdf_neg_);
    dctx.cnt_pos   = (int)sub_pos_.size();
    dctx.cnt_neg   = (int)sub_neg_.size();

    dctx.icdf_kernel = raw_ptr(icdf_kernel_);
    dctx.kernel_L    = (int)icdf_kernel_.size();
    dctx.h_w         = P_.h_w;

    dctx.cum_pos   = cum_pos_;
    dctx.cum_neg   = cum_neg_;
    dctx.Lx = P_.Lx; dctx.Ly = P_.Ly;
    dctx.InvLx = 1.0f / P_.Lx; dctx.InvLy = 1.0f / P_.Ly;
    dctx.periodic  = periodic;

    dctx.dx = P_.dx;

    dctx.pos_view = makePosView();
    dctx.neg_view = makeNegView();

    dvec<float2> velC_pos(N), velL_pos(N), velR_pos(N), velT_pos(N), velB_pos(N);
    dvec<float2> velC_neg(N), velL_neg(N), velR_neg(N), velT_neg(N), velB_neg(N);
    cudaMemset(raw_ptr(velC_pos), 0, N*sizeof(float2));
    cudaMemset(raw_ptr(velL_pos), 0, N*sizeof(float2));
    cudaMemset(raw_ptr(velR_pos), 0, N*sizeof(float2));
    cudaMemset(raw_ptr(velT_pos), 0, N*sizeof(float2));
    cudaMemset(raw_ptr(velB_pos), 0, N*sizeof(float2));
    cudaMemset(raw_ptr(velC_neg), 0, N*sizeof(float2));
    cudaMemset(raw_ptr(velL_neg), 0, N*sizeof(float2));
    cudaMemset(raw_ptr(velR_neg), 0, N*sizeof(float2));
    cudaMemset(raw_ptr(velT_neg), 0, N*sizeof(float2));
    cudaMemset(raw_ptr(velB_neg), 0, N*sizeof(float2));

    monte_carlo_bs_kernel_safe<<<grid, TPB>>>(
        particles_, dctx, P_.N1, P_.N2, /*sign=*/+1, /*aux=*/1,
        raw_ptr(velC_pos), raw_ptr(velL_pos), raw_ptr(velR_pos),
        raw_ptr(velT_pos), raw_ptr(velB_pos));
    monte_carlo_bs_kernel_safe<<<grid, TPB>>>(
        particles_, dctx, P_.N1, P_.N2, /*sign=*/-1, /*aux=*/1,
        raw_ptr(velC_neg), raw_ptr(velL_neg), raw_ptr(velR_neg),
        raw_ptr(velT_neg), raw_ptr(velB_neg));
    // CUDA_CHECK();  //

    dvec<float2> velC(N), velL(N), velR(N), velT(N), velB(N);
    auto combine = [] __device__ (float2 a, float2 b){ return make_float2(a.x-b.x, a.y-b.y); };
    thrust::transform(velC_pos.begin(), velC_pos.end(), velC_neg.begin(), velC.begin(), combine);
    thrust::transform(velL_pos.begin(), velL_pos.end(), velL_neg.begin(), velL.begin(), combine);
    thrust::transform(velR_pos.begin(), velR_pos.end(), velR_neg.begin(), velR.begin(), combine);
    thrust::transform(velT_pos.begin(), velT_pos.end(), velT_neg.begin(), velT.begin(), combine);
    thrust::transform(velB_pos.begin(), velB_pos.end(), velB_neg.begin(), velB.begin(), combine);

    update_jacobian_from5<<<(N+TPB-1)/TPB, TPB>>>(
        N, P_.dt, P_.dx,
        raw_ptr(velC), raw_ptr(velL), raw_ptr(velR),
        raw_ptr(velT), raw_ptr(velB),
        raw_ptr(particles_.F));
    // CUDA_CHECK();

    // 6)
    particles_.vel = velC;          //
    thrust::transform(particles_.pos.begin(), particles_.pos.end(),
                      velC.begin(), particles_.pos.begin(),
                      [dt=P_.dt] __device__ (float2 p, float2 v){
                          return make_float2(p.x + v.x*dt, p.y + v.y*dt);
                      });
    if (periodic){
        thrust::for_each(particles_.pos.begin(), particles_.pos.end(),
            [Lx=P_.Lx, Ly=P_.Ly] __device__ (float2& p){
                p.x -= floorf(p.x / Lx) * Lx;
                p.y -= floorf(p.y / Ly) * Ly;
            });
    }
}

void Simulation::compute_velocity_naive(bool periodic, float eps_core)
{
    const int N   = particles_.N_cur;
    dim3 grid((N + TPB - 1) / TPB);

    particles_.vel.resize(N);

    biot_savart_naive_kernel<<<grid, TPB>>>(
        raw_ptr(particles_.pos),
        raw_ptr(particles_.omega),   //
        N,
        raw_ptr(particles_.vel),
        P_.Lx, P_.Ly, periodic,
        eps_core);
    cudaDeviceSynchronize();

}

void Simulation::step_cuda_naive(bool periodic)
{
    const float eps_core = 0.5f * P_.h_w;   //
    compute_velocity_naive(periodic, eps_core);

    thrust::transform(particles_.pos.begin(), particles_.pos.end(),
                      particles_.vel.begin(),
                      particles_.pos.begin(),
                      [dt=P_.dt] __device__ (float2 p, float2 v){
                        return make_float2(p.x + v.x*dt, p.y + v.y*dt);
                      });

    if (periodic){
        thrust::for_each(particles_.pos.begin(), particles_.pos.end(),
                         [Lx=P_.Lx, Ly=P_.Ly] __device__ (float2& p){
                            p.x -= floorf(p.x / Lx) * Lx;
                            p.y -= floorf(p.y / Ly) * Ly;
                         });
    }
}
