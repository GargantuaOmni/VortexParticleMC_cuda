#include "cg_rbf_init.cuh"
#include "device_vector_alias.hpp"
#include "vortex_particle_mc.hpp"
#include "vorticity_query.cuh"
#include "utilities.hpp"
#include <thrust/sequence.h>
#include <thrust/inner_product.h>
#include <thrust/transform.h>
#include <thrust/logical.h>

__global__ void matvec_kernel(
        ParticleSet          ps,
        VorticityView        viewAll,
        float                h_w,
        bool                 periodic,
        const float*   __restrict__ x,
        float*          __restrict__ y)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= ps.N_cur) return;

    viewAll.w = x;
    float2 p = ps.d_p()[gid];

    y[gid] = queryVorticityAbs(viewAll, p,
                               /*sign=*/0,
                               /*use_signed=*/true,
                               /*exclude_center=*/false,
                               h_w,
                               periodic);
}

void Ax_cuda(const Simulation& sim,
             const float*      x,
             float*            y,
             cudaStream_t      stream)
{
    auto view = sim.makeAllView(x);
    int threads = 256;
    int blocks  = (sim.particles_.N_cur + threads - 1) / threads;


    matvec_kernel<<<blocks, threads, 0, stream>>>(
        sim.particles_,
        view,
        sim.P_.h_w,
        sim.P_.periodic,
        x, y);

}


void conjugate_gradient_rbf(Simulation& sim,
                            float tol,
                            int   max_iter)
{
    std::cout << "onjugate_gradient_rbf starts..." << std::endl;
    const int N = sim.particles_.N_cur;
    dvec<float> w(N), r(N), p(N), Ap(N), Aw(N);

    // Debugging...
    if (sim.particles_.N_cur == 0)
        throw std::runtime_error("CG init: N_cur == 0 (particles not generated)");
    if (thrust::all_of(sim.particles_.omega_field.begin(),
                       sim.particles_.omega_field.end(),
                       [] __device__ (float w){ return w == 0.f; }))
        std::cerr << "[Warning] omega_field is all zero, CG will return w = 0.\n";
    // *************************************************************************   //

    float diag = cubicSplinePDF(0.f, sim.P_.h_w);
    thrust::transform(sim.particles_.omega_field.begin(),
                      sim.particles_.omega_field.end(),
                      w.begin(),
                      [diag] __device__(float b){ return b / diag; });

    sim.build_global_hash();

    Ax_cuda(sim, raw_ptr(w), raw_ptr(Aw));
    thrust::transform(sim.particles_.omega_field.begin(),
                      sim.particles_.omega_field.end(),
                      Aw.begin(), r.begin(),
                      thrust::minus<float>());
    p = r;

    float r2_old = thrust::inner_product(r.begin(), r.end(), r.begin(), 0.f);

    for (int k = 0; k < max_iter; ++k)
    {
        Ax_cuda(sim, raw_ptr(p), raw_ptr(Ap));

        float dot_pAp = thrust::inner_product(p.begin(), p.end(), Ap.begin(), 0.f);
        float alpha   = r2_old / dot_pAp;

        thrust::transform(w.begin(), w.end(), p.begin(), w.begin(),
             [alpha] __device__(float wi, float pi){ return wi + alpha * pi; });
        thrust::transform(r.begin(), r.end(), Ap.begin(), r.begin(),
             [alpha] __device__(float ri, float Api){ return ri - alpha * Api; });

        float r2_new = thrust::inner_product(r.begin(), r.end(), r.begin(), 0.f);
        if (sqrtf(r2_new) /
            sqrtf(thrust::inner_product(sim.particles_.omega_field.begin(),
                                        sim.particles_.omega_field.end(),
                                        sim.particles_.omega_field.begin(), 0.f))
            < tol) break;

        float beta = r2_new / r2_old;
        thrust::transform(r.begin(), r.end(), p.begin(), p.begin(),
             [beta] __device__(float ri, float pi){ return ri + beta * pi; });
        r2_old = r2_new;
        std::cout << "conjugate_gradient_rbf run " << k << " steps" << std::endl;
    }

    thrust::copy(w.begin(), w.end(), sim.particles_.omega.begin());
    sim.build_subsets_cuda();
    sim.do_spatial_hashing();
}