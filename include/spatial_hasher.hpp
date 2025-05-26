#pragma once
#include "vortex_particle_mc.hpp"
#include <vector>
#include <glm/vec2.hpp>

enum class Backend { CPU, OMP, CUDA };

class SpatialHasher {
public:
    SpatialHasher(int rshx, int rshy, Backend b);

    void resize(int N_max);       //
    void build(const std::vector<glm::vec2>& pos,
               const std::vector<int>&       sub,   /*  */
               float Lx, float Ly);           /* vp_cell / cell_num */

    const SpatialHash& data() const { return hash_; }

private:
    SpatialHash hash_;
    Backend     backend_;
};
