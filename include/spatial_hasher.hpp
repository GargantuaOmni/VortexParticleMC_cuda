#pragma once
#include "vortex_particle_mc.hpp"
#include <vector>
#include <glm/vec2.hpp>

enum class Backend { CPU, OMP, CUDA };

class SpatialHasher {
public:
    SpatialHasher(int rshx, int rshy, Backend b);

    void resize(int N_max);       //
    template<typename VecPos, typename VecSub>
    void build(const VecPos& pos, const VecSub& sub, float Lx, float Ly);

    const SpatialHash& data() const { return hash_; }

private:
    SpatialHash hash_;
    Backend     backend_;
};
