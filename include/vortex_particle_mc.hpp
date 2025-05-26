//include/vortex_particle_mc.h

#pragma once
#include <span>
#include <vector>
#include <glm/vec2.hpp>


struct ParticleSet {
    int N_max  = 0;
    int N_cur  = 0;
    std::vector<glm::vec2> pos;
    std::vector<float>     omega;          // vorticity
};

struct SpatialHash {
    int rshx = 0;
    int rshy = 0;

    std::vector<int> cell_num;
    std::vector<int> cell_acc;

    std::vector<int> vp_cell;
    std::vector<int> vp_sort;

    SpatialHash() = default;
    SpatialHash(int rx,int ry) : rshx(rx), rshy(ry)
    {
        int cells = rx*ry;
        cell_num.resize(cells);
        cell_acc.resize(cells);
    }

    void resize_particles(int N_max)
    {
        vp_cell.resize(N_max);
        vp_sort.resize(N_max);
    }
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
