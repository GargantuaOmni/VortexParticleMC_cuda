//include/vortex_particle_mc.hpp

#pragma once
#include <span>
#include <vector>
#include <cassert>
#include <cmath>
#include <random>
#include <device_vector_alias.hpp>
#include "spatial_hasher.hpp"
#include <spatial_hash.hpp>
#include <iostream>
//#include <glm/vec2.hpp>


struct ParticleSet
{
    int N_max {0};
    int N_cur {0};

    // Buffers
    dvec<float2> pos;
    dvec<float2> pos_left, pos_right, pos_top, pos_bottom;
    dvec<float2> pos_temp;
    dvec<float2> vel;

    dvec<float> omega;      //
    dvec<float> omega_field;      //
    dvec<float> jac;        // Jacobian

    /* ----  ---- */
    void resize(int cap)
    {
        N_max = cap;
        N_cur = cap;              // Initialize all particles as active, however, we should set N_cur when we really seed particles

        pos.resize(cap);
        vel.resize(cap);
        pos_left.resize(cap);
        pos_right.resize(cap);
        pos_top.resize(cap);
        pos_bottom.resize(cap);
        pos_temp.resize(cap);
        omega.resize(cap);
        omega_field.resize(cap);
        jac.resize(cap);
    }

    void set_n_current(const int N_cur_) {
        N_cur = N_cur_;
    }

    /* ---- GPU kernel raw pointer getter ---- */
    __host__ __device__ float2* d_p()       { return raw_ptr(pos); }
    __host__ __device__ float2* d_pt()       { return raw_ptr(pos_temp); }
    __host__ __device__ float2* d_v()       { return raw_ptr(vel); }

    __host__ __device__ float* d_omega()    { return raw_ptr(omega); }
    __host__ __device__ float* d_omega_field()    { return raw_ptr(omega_field); }
    __host__ __device__ float* d_jac()      { return raw_ptr(jac); }

    __host__ __device__ const float2* d_p()     const { return raw_ptr(pos); }
    __host__ __device__ const float2* d_pt()    const { return raw_ptr(pos_temp); }
    __host__ __device__ const float2* d_v()     const { return raw_ptr(vel); }

    __host__ __device__ const float*  d_omega() const { return raw_ptr(omega); }
    __host__ __device__ const float*  d_omega_field() const { return raw_ptr(omega_field); }
    __host__ __device__ const float*  d_jac()   const { return raw_ptr(jac); }

};


struct VorticityView
{
    /* Existing */
    const float2* pos;
    const float*  w;
    int           N;

    /* Hashing */
    const int* vp_sort;          // local id
    const int* cell_acc;
    int        rshx{}, rshy{};
    float      Lx, Ly;
    int        cnt_view;

    const int* sub;              // local id -> global id
};

struct SimParam
{
    /* Parameters */
    float dt          = 0.001f;
    float Lx          = 1.f;
    float Ly          = 1.f;
    int   ResolutionX = 200;   // Nx
    float hwr         = 8.f;   // = h_w / h
    int   N_max       = 4000;
    int N1          = 1000;
    int N2          = 1000;

    /* Parameters for later computations */
    int   ResolutionY = 0;     // Ny
    float h          = 0.f;    // h = Lx / ResolutionX
    float h_w        = 0.f;    // h_w = hwr * h
    int length_iCDF  = 10;

    /* ------------ Construction ------------ */
    explicit SimParam(float dt_ = 0.00001f,
                      float Lx_ = 1.f, float Ly_ = 1.f,
                      int   resX = 200,
                      float hwr_ = 12.f)
    : dt(dt_), Lx(Lx_), Ly(Ly_), ResolutionX(resX), hwr(hwr_)
    {
        assert(ResolutionX > 0 && "ResolutionX must be positive");

        h  = Lx / float(ResolutionX);             //
        ResolutionY = int(std::round(Ly / h));
        if (ResolutionY == 0)  ResolutionY = 1;    //

        h_w = h * hwr;                     // support-radius
    }
};

class Simulation {
public:

    ParticleSet particles_;
    SimParam   P_;

    explicit Simulation(SimParam p);

    void init(int num_of_p);     //

    void do_spatial_hashing();

    void build_subsets_cpu();
    void build_subsets_cuda();

    std::mt19937 rng;
    void step_cpu(bool periodic);
    void step_cuda(bool periodic);

    inline float eval_vorticity_cpu(float2 p, bool periodic) const;


    /* --- Test functions --- */
    void test_vorticity_grid(int res);
    /* --- Test functions ---  */

    VorticityView makePosView(const float* w_ptr  = nullptr) const;
    VorticityView makeNegView(const float* w_ptr  = nullptr) const;
    VorticityView makePosViewNaive(const float* w_ptr = nullptr) const;
    VorticityView makeNegViewNaive(const float* w_ptr = nullptr) const;

private:

    SpatialHash hash_pos_, hash_neg_;
    dvec<int>   sub_pos_, sub_neg_; // Sub-index for the positive and negative particles
    dvec<float> cdf_pos_, cdf_neg_; // cdf
    int         cnt_pos_{}, cnt_neg_{};
    float       cum_pos_{}, cum_neg_{};

    dvec<float> cdf_kernel, icdf_kernel; // cdf and icdf for kernels

    VorticityView pos_view;
    VorticityView neg_view;
};


float evalVorticityAbs_cpu(const VorticityView& v,
                           const float2& p,
                           int sign,                   //
                           bool use_signed,
                           bool exclude_center,
                           float h_w,
                           bool periodic);


inline float Simulation::eval_vorticity_cpu(float2 p, bool periodic) const
{
    float w_pos = evalVorticityAbs_cpu(pos_view, p, 1, false, false, P_.h_w, periodic);
    float w_neg = evalVorticityAbs_cpu(neg_view, p, -1, false, false, P_.h_w, periodic);
    return w_pos - w_neg;
}