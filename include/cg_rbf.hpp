#pragma once
#include "vortex_particle_mc.hpp"     // Simulation / ParticleSet / SpatialHash

void build_global_hash_all(Simulation& sim);

void solve_cg_rbf_cpu(Simulation& sim,
                      float h_w,
                      float rel_tol   = 3e-3f,
                      float alpha_tol = 1e-8f,
                      int   max_iter  = 1000);

