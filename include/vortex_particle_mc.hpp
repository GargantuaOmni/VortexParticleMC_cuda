//include/vortex_particle_mc.h

#pragma once
#include <span>
#include <vector>
#include <device_vector_alias.hpp>
#include <spatial_hash.hpp>
#include <glm/vec2.hpp>


struct ParticleSet {
    int N_max  = 0;
    int N_cur  = 0;
    std::vector<glm::vec2> pos;
    std::vector<float>     omega;          // vorticity
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

    void init();     //
    void step();     //

private:
    SimParam   P_;
    ParticleSet particles_;
    SpatialHash hash_pos_, hash_neg_;
    std::vector<int> sub_pos_, sub_neg_;   // Sub map that divides all particles
};
