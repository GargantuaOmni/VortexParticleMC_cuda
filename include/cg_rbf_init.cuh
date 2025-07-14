#pragma once
#include "vortex_particle_mc.hpp"
#include "vorticity_query.cuh"   // queryVorticityAbs
#include "device_vector_alias.hpp"

void Ax_cuda(const Simulation&  sim,
                 const float*       x,
                 float*             y,
                 cudaStream_t       stream = 0);

void conjugate_gradient_rbf(Simulation& sim,
                            float tol        = 5e-3f,
                            int   max_iter   = 1000);