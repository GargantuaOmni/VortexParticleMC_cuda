//include/vortex_particle_mc.hpp

#pragma once
#include <span>
#include <vector>
#include <device_vector_alias.hpp>
#include "spatial_hasher.hpp"
#include <spatial_hash.hpp>
//#include <glm/vec2.hpp>


struct ParticleSet
{
    int N_max {0};
    int N_cur {0};

    // Buffers
    dvec<float2> pos;
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
        pos_temp.resize(cap);
        omega.resize(cap);
        omega_field.resize(cap);
        jac.resize(cap);
    }

    /* ---- GPU kernel raw pointer getter ---- */
    float2* d_p()       { return raw_ptr(pos); }
    float2* d_pt()       { return raw_ptr(pos_temp); }
    float2* d_v()       { return raw_ptr(vel); }

    float* d_omega()    { return raw_ptr(omega); }
    float* d_omega_field()    { return raw_ptr(omega_field); }
    float* d_jac()      { return raw_ptr(jac); }

    const float2* d_p() const { return raw_ptr(pos); }
};

struct SimParam {
    float dt  = 0.1f;
    float Lx  = 1.f, Ly = 1.f;
    int   N_max = 80000;
    int   ResolutionX = 200;
    float hwr = 4.f;         // h_w / h
};

class Simulation {
public:
    explicit Simulation(SimParam p);

    void init(int num_of_p);     //
    void step();     //

    void do_spatial_hashing();

    void build_subsets_cpu();
    void build_subsets_cuda();

private:
    SimParam   P_;
    ParticleSet particles_;
    SpatialHash hash_pos_, hash_neg_;
    dvec<int>   sub_pos_, sub_neg_; // Sub-index for the positive and negative particles
    dvec<float> cdf_pos_, cdf_neg_; // cdf
    int         cnt_pos_{}, cnt_neg_{};
    float       cum_pos_{}, cum_neg_{};
};
