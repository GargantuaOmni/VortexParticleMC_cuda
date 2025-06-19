#include "cg_rbf.hpp"
#include "utilities.hpp"          // cubicSplinePDF
#include "analytical_flow.hpp"    // taylor_green_vorticity
#include "evaluate_vorticity.hpp" // evalVorticityAbs_cpu
#include <vector>
#include <numeric>
#include <cmath>
#include <iostream>
#include <omp.h>

void build_global_hash_all(Simulation& sim)
{
    int N = sim.particles_.N_cur;

    sim.sub_all_.resize(N);
    std::iota(sim.sub_all_.begin(), sim.sub_all_.end(), 0);

    int rshx = sim.hash_pos_.rshx;
    int rshy = sim.hash_pos_.rshy;
    sim.hash_all_.resize_cells(rshx, rshy);
    sim.hash_all_.resize_particles(N);

    /* 拆 pos.x/y */
    std::vector<float> px(N), py(N);
    for(int i=0;i<N;++i){ px[i]=sim.particles_.pos[i].x;
                          py[i]=sim.particles_.pos[i].y; }

    FillCells_cpu(rshx,rshy, sim.P_.Lx, sim.P_.Ly, N,
                  sim.sub_all_,
                  px, py,
                  sim.hash_all_.cell_num,
                  sim.hash_all_.cell_acc,
                  sim.hash_all_.vp_cell);

    CountingSort_cpu(N,
                     sim.hash_all_.cell_num,
                     sim.hash_all_.cell_acc,
                     sim.hash_all_.vp_cell,
                     sim.hash_all_.vp_sort);
}

/* ---------- 2. A·w : 直接 evalVorticityAbs_cpu(sign=0) ---------- */
static void apply_kernel(const Simulation& sim,
                         const VorticityView& view_all,
                         const std::vector<float>& w,
                         std::vector<float>&       Aw,
                         float h_w)
{
    int N = sim.particles_.N_cur;
    Aw.resize(N);

    VorticityView vw = view_all;               //
    vw.w = const_cast<float*>(w.data());       //

    #pragma omp parallel for
    for(int i=0;i<N;++i){
        float2 p = sim.particles_.pos[i];
        Aw[i] = evalVorticityAbs_cpu(vw, p,
                                     /*sign=*/0,
                                     /*use_signed=*/false,
                                     /*exclude_center=*/true,
                                     h_w);
    }
}

static float dot(const std::vector<float>& a,
                 const std::vector<float>& b)
{
    float s=0.f;
    #pragma omp parallel for reduction(+:s)
    for(int i=0;i<int(a.size());++i) s+=a[i]*b[i];
    return s;
}
static void axpy(float a,
                 const std::vector<float>& x,
                 std::vector<float>&       y)
{
    #pragma omp parallel for
    for(int i=0;i<int(x.size());++i) y[i]+=a*x[i];
}

void solve_cg_rbf_cpu(Simulation& sim,
                      float h_w, float rel_tol,
                      float alpha_tol, int max_iter)
{
    int N = sim.particles_.N_cur;
    build_global_hash_all(sim);

    VorticityView view_all;
    view_all.pos   = raw_ptr(sim.particles_.pos);
    view_all.w     = raw_ptr(sim.particles_.omega);    //
    view_all.vp_sort   = raw_ptr(sim.hash_all_.vp_sort);
    view_all.vp_cell   = raw_ptr(sim.hash_all_.vp_cell);
    view_all.cell_acc  = raw_ptr(sim.hash_all_.cell_acc);
    view_all.N   = N;
    view_all.rshx= sim.hash_all_.rshx;
    view_all.rshy= sim.hash_all_.rshy;

    std::vector<float> b(N);
    for(int i=0;i<N;++i){
        auto p = sim.particles_.pos[i];
        b[i] = taylor_green_vorticity(p.x, p.y);
    }

    std::vector<float>& w = sim.particles_.omega;
    w.assign(N,0.f);                       // 0

    std::vector<float> r=b, p=r, Ap(N);

    float b_norm = std::sqrt(dot(b,b));
    float rs_old = dot(r,r);

    for(int it=0; it<max_iter; ++it)
    {
        apply_kernel(sim, view_all, p, Ap, h_w);

        float denom = dot(p,Ap);
        if(std::fabs(denom) < 1e-12f){ std::cerr<<"|pAp| too small\n"; break; }
        float alpha = rs_old / denom;

        if(std::fabs(alpha) < alpha_tol){
            std::cout<<"alpha too small, stop\n"; break;
        }

        axpy(alpha, p, w);        // w += αp
        axpy(-alpha,Ap, r);       // r -= αAp

        float rs_new = dot(r,r);
        if(std::sqrt(rs_new) < rel_tol*b_norm){
            std::cout<<"CG converged "<<it+1
                     <<" iters  rel-res "<<std::sqrt(rs_new)/b_norm<<"\n";
            return;
        }
        float beta = rs_new / rs_old;
        #pragma omp parallel for
        for(int i=0;i<N;++i) p[i]=r[i]+beta*p[i];
        rs_old = rs_new;
    }
    std::cout<<"CG max_iter reached, rel-res "
             <<std::sqrt(rs_old)/b_norm<<"\n";
}
