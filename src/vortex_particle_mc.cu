//src/vortex_particle_mc.cu
#include <vortex_particle_mc.hpp>
#include <utilities.hpp>
#include <utility>
#include <cstdio>
#include <random>
#include <iostream>
#include <iomanip>
#include "viz_utils.hpp"
#include <iomanip>
#include <sstream>
#include <chrono>
#include <numeric>


#if defined(USE_CUDA) && defined(__CUDACC__)
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/count.h>
#include <vortex_particle_mc.hpp>
#include <vorticity_query.cuh>
#endif

#include "stb_image_write.h"
#include "analytical_flow.hpp"
#include "monte_carlo_bs.hpp"

int main()
{
    std::filesystem::create_directories("result/vel");

    /* ---------- 初始化 ---------- */
    SimParam    param;
    Simulation  vp_mc_simulation(param);
    vp_mc_simulation.init(param.N_max);

    const int NSTEPS = 3;
    const int PNG_RES = 512;

    //dump_vorticity_png(vp_mc_simulation, "result/step_init.png", PNG_RES, true);

    for(int step = 0; step < NSTEPS; ++step)
    {
#if defined(USE_CUDA) && defined(__CUDACC__)
        /* ---- GPU ---- */
        vp_mc_simulation.step_cuda(/*periodic=*/true);
#else
        /* ---- CPU ---- */
        vp_mc_simulation.step_cpu(/*periodic=*/true);
#endif
        vp_mc_simulation.do_spatial_hashing();

        std::ostringstream oss_png, oss_svg;
        oss_png << "result/step_" << std::setw(4) << std::setfill('0') << step << ".png";
        oss_svg << "result/vel/vel_step_" << std::setw(4) << std::setfill('0') << step << ".svg";

        dump_vorticity_png(vp_mc_simulation, oss_png.str(), PNG_RES, /*periodic=*/true);
        dump_velocity_svg_centered(vp_mc_simulation, oss_svg.str(),
                                   /*vel_scale=*/0.25f, /*periodic=*/true);

        std::cout << "frame " << step << "  ->  " << oss_png.str() << '\n';
    }

    return 0;
}

Simulation::Simulation(SimParam p) : P_(std::move(p)) {}

void Simulation::init(int num_of_p)
{
    assert(num_of_p > 0 && num_of_p <= P_.N_max);
    particles_.resize(P_.N_max);
    particles_.set_n_current(num_of_p);

    /* ---- 1. 生成随机粒子，赋 Taylor–Green ω ---- */
    std::vector<float2> host_pos(num_of_p);
    std::vector<float>  host_omega(num_of_p);

    std::mt19937 rng(42);                                   //
    std::uniform_real_distribution<float> uni(0.f, 1.f);   // U[0,1)

    for(int i = 0; i < num_of_p; ++i){
        float2 p { uni(rng), uni(rng) };      //
        host_pos[i]   = p;
        host_omega[i] = taylor_green_vorticity(p);          //
    }

#if defined(USE_CUDA) && defined(__CUDACC__)
    thrust::copy(host_pos.begin(),   host_pos.end(),   particles_.pos.begin());
    thrust::copy(host_omega.begin(), host_omega.end(), particles_.omega.begin());
    std::cout << "particles_.omega.size()    "  << particles_.omega.size() << std::endl;

    std::cout << "Try to invoke subsets cuda building..." << std::endl;
    build_subsets_cuda();
#else
    particles_.pos   = host_pos;
    particles_.omega = host_omega;
    build_subsets_cpu();
#endif
    std::cout << "Try to invoke spatial hashing cuda building..."  << std::endl;
    do_spatial_hashing();
    std::cout << "Spatial Hashing finished..."  << std::endl;
    std::cin.get();
}


void Simulation::do_spatial_hashing()
{
    int rshx = P_.ResolutionX / int(P_.hwr) / 2;      // The same as Taichi
    int rshy = P_.ResolutionY / int(P_.hwr) / 2;

    /* ---------- Resize the forms ---------- */
    hash_pos_.resize_cells(rshx, rshy);
    hash_neg_.resize_cells(rshx, rshy);

    hash_pos_.resize_particles(int(sub_pos_.size()));
    hash_neg_.resize_particles(int(sub_neg_.size()));

#if defined(USE_CUDA)
    const float2* d_p = particles_.d_p();
    const int*   d_sub_pos = raw_ptr(sub_pos_);
    const int*   d_sub_neg = raw_ptr(sub_neg_);

    FillCells_cuda(rshx,rshy,P_.Lx,P_.Ly,
                   int(sub_pos_.size()),
                   d_sub_pos, d_p,
                   raw_ptr(hash_pos_.cell_num),
                   raw_ptr(hash_pos_.cell_acc),
                   raw_ptr(hash_pos_.vp_cell));

    CountingSort_cuda(int(sub_pos_.size()),
                      raw_ptr(hash_pos_.cell_num),
                      raw_ptr(hash_pos_.cell_acc),
                      raw_ptr(hash_pos_.vp_cell),
                      raw_ptr(hash_pos_.vp_sort));

    FillCells_cuda(rshx,rshy,P_.Lx,P_.Ly,
                   int(sub_neg_.size()),
                   d_sub_neg, d_p,
                   raw_ptr(hash_neg_.cell_num),
                   raw_ptr(hash_neg_.cell_acc),
                   raw_ptr(hash_neg_.vp_cell));

    CountingSort_cuda(int(sub_neg_.size()),
                      raw_ptr(hash_neg_.cell_num),
                      raw_ptr(hash_neg_.cell_acc),
                      raw_ptr(hash_neg_.vp_cell),
                      raw_ptr(hash_neg_.vp_sort));

#else
    static std::vector<float> px, py;
    auto split = [&](const dvec<float2>& pos){
        size_t N = pos.size(); px.resize(N); py.resize(N);
        for(size_t i=0;i<N;++i){ px[i]=pos[i].x; py[i]=pos[i].y; }
    };
    split(particles_.pos);

    auto run_cpu = [&](const dvec<int>&  sub,
                       SpatialHash&      H)
    {
        FillCells_cpu(rshx,rshy,P_.Lx,P_.Ly,
                      int(sub.size()),
                      sub,
                      std::span<const float>(px.data(), particles_.N_cur),
                      std::span<const float>(py.data(), particles_.N_cur),
                      H.cell_num, H.cell_acc, H.vp_cell);
        auto cell_num_copy = H.cell_num;
        CountingSort_cpu(int(sub.size()),
                         H.cell_num, H.cell_acc,
                         H.vp_cell,  H.vp_sort);

        int sum = std::accumulate(H.cell_num.begin(), H.cell_num.end(), 0);
        assert(sum==0);

        int C = H.rshx * H.rshy;
        for(int c=0; c<C; ++c){
            int beg = H.cell_acc[c];
            int end = (c+1<C) ? H.cell_acc[c+1] : H.vp_sort.size()-1;
            for(int k=beg; k<end; ++k){
                int local = H.vp_sort[k];
                assert(H.vp_cell[local] == c);
            }
        }

    };

    run_cpu(sub_pos_, hash_pos_);
    run_cpu(sub_neg_, hash_neg_);
#endif

    pos_view = makePosView();
    neg_view = makeNegView();

}

VorticityView Simulation::makePosView(const float* w_ptr) const
{
    const SpatialHash& H = hash_pos_;    //
    return { particles_.d_p(),
             w_ptr ? w_ptr : particles_.d_omega(),
             particles_.N_cur,
             raw_ptr(H.vp_sort),         // Local counting sort
             raw_ptr(H.cell_acc),
             H.rshx, H.rshy,
             P_.Lx, P_.Ly,cnt_pos_, raw_ptr(sub_pos_) };        //
}


VorticityView Simulation::makeNegView(const float* w_ptr) const
{
    const SpatialHash& H = hash_neg_;    //
    return { particles_.d_p(),
             w_ptr ? w_ptr : particles_.d_omega(),
             particles_.N_cur,
             raw_ptr(H.vp_sort),         // Local counting sort
             raw_ptr(H.cell_acc),
             H.rshx, H.rshy,
              P_.Lx, P_.Ly,cnt_neg_, raw_ptr(sub_neg_) };        //
}

VorticityView Simulation::makePosViewNaive(const float* w_ptr) const
{
    return { particles_.d_p(), w_ptr ? w_ptr : particles_.d_omega(), particles_.N_cur,
             nullptr, nullptr, 0, 0,
        P_.Lx, P_.Ly,cnt_pos_, raw_ptr(sub_pos_) };
}
VorticityView Simulation::makeNegViewNaive(const float* w_ptr) const
{
    return { particles_.d_p(), w_ptr ? w_ptr : particles_.d_omega(), particles_.N_cur,
             nullptr, nullptr, 0, 0,
        P_.Lx, P_.Ly,cnt_neg_, raw_ptr(sub_neg_) };
}


void Simulation::test_vorticity_grid(int res /*=128*/)
{

    const float h_w = P_.h_w;
    const float step = 1.f / res;

#if defined(USE_CUDA)
    auto viewPosHash_d  = makePosView();
    auto viewPosNaive_d = makePosViewNaive();

    dvec<float> d_field_hash(res*res);
    dvec<float> d_field_naive(res*res);

    float* hash_ptr  = thrust::raw_pointer_cast(d_field_hash.data());
    float* naive_ptr = thrust::raw_pointer_cast(d_field_naive.data());

    auto kernel = [=] __device__ (int idx){
        int ix = idx / res, iy = idx % res;
        float2 p = {(ix+0.5f)*step, (iy+0.5f)*step};
        float v_h = queryVorticityAbs(viewPosHash_d , p, +1,false,false,h_w, false);
        float v_n = queryVorticityAbsNaive(viewPosNaive_d, p, +1,false,false,h_w);
        hash_ptr[idx]  = v_h;
        naive_ptr[idx] = v_n;
    };

    std::cout << "Debug 10" << std::endl;

    thrust::for_each(thrust::device,
                     thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(res*res),
                     kernel);

    std::vector<float> field_hash(res * res);
    std::vector<float> field_naive(res * res);

    std::cout << "Debug 11" << std::endl;

    thrust::copy(d_field_hash.begin(), d_field_hash.end(), field_hash.begin());
    thrust::copy(d_field_naive.begin(), d_field_naive.end(), field_naive.begin());

    std::cout << "Debug 12" << std::endl;

    double max_abs_err=0, sum_rel_err=0;
    for(size_t i=0;i<field_hash.size();++i){
        double abs_err = std::fabs(field_hash[i]-field_naive[i]);
        double rel_err = abs_err / (std::fabs(field_naive[i])+1e-12);
        max_abs_err = std::max(max_abs_err,abs_err);
        sum_rel_err+= rel_err;
    }

    std::cout << "Debug 13" << std::endl;
#else
    auto viewPosHash   = makePosView();
    auto viewNegHash   = makeNegView();
    auto viewPosNaive  = makePosViewNaive();
    auto viewNegNaive  = makeNegViewNaive();


    std::vector<float> field_hash(res*res);
    std::vector<float> field_naive(res*res);

    double max_abs_err = 0.0, sum_rel_err = 0.0;//
    for(int ix=0; ix<res; ++ix)
        for(int iy=0; iy<res; ++iy)
        {
            float2 p = { (ix+0.5f)*step, (iy+0.5f)*step };

            float v_hash =
                  evalVorticityAbs_cpu(viewPosHash, p, +1, false, true, h_w, false);
                  // - evalVorticityAbs_cpu(viewNegHash, p, -1, false, true, h_w);

            float v_naive =
                  evalVorticityAbs_cpu(viewPosNaive, p, +1, false, true, h_w, false);
                  // - evalVorticityAbs_cpu(viewNegNaive, p, -1, false, true, h_w);

            field_hash[ix*res + iy]  = v_hash;
            field_naive[ix*res + iy] = v_naive;

            double abs_err = std::fabs(v_hash - v_naive);
            double rel_err = abs_err / (std::fabs(v_naive) + 1e-12);

            max_abs_err   = std::max(max_abs_err, abs_err);
            sum_rel_err  += rel_err;
        }//
    for(int i = 0; i < particles_.N_cur; ++i)
    {
        float2 p = particles_.pos[i];
        float v_hash  = evalVorticityAbs_cpu(makePosView(),      p, +1,false,false,P_.h_w, false);
        float v_naive = evalVorticityAbs_cpu(makePosViewNaive(), p, +1,false,false,P_.h_w, false);

        /*std::cout << std::setw(3) << i
                  << "  hash="  << std::scientific << v_hash
                  << "  naive=" << v_naive << '\n';
                  */
    }
#endif
    auto dump_png = [&](const std::vector<float>& f, const char* name){
        float vmin=*std::min_element(f.begin(), f.end());
        float vmax=*std::max_element(f.begin(), f.end());
        float inv  = (vmax-vmin>1e-9f)? 1.f/(vmax-vmin):1.f;

        std::vector<unsigned char> img(res*res*3);
        for(size_t i=0;i<f.size();++i){
            float norm = (f[i]-vmin)*inv;     // 0~1
            unsigned char r=0,g=0,b=0;
            if(norm>=0.5f){                   // [0.5,1] red
                r = (unsigned char)(255*(norm-0.5f)*2);
            }else{                            // [0,0.5) blue
                b = (unsigned char)(255*(0.5f-norm)*2);
            }
            g = (unsigned char)(255*norm);    //
            img[3*i+0]=r; img[3*i+1]=g; img[3*i+2]=b;
        }
        stbi_write_png(name,res,res,3,img.data(),res*3);
    };

    dump_png(field_hash , "vort_hash.png");
    dump_png(field_naive, "vort_naive.png");

    /* ---------- 误差报告 ---------- */
    double mean_rel = sum_rel_err / (res*res);
    std::cout << "[Vorticity Grid Test " << res << "×" << res << "]\n"
              << "  max|err|  = " << max_abs_err  << '\n'
              << "  mean rel  = " << mean_rel     << '\n'
              << "  PNG saved → vort_hash.png / vort_naive.png\n\n";
}

