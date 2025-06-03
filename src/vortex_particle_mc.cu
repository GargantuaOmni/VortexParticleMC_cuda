#include <vortex_particle_mc.hpp>
#include <utility>
#include <cstdio>
#include <random>
#include <cstdio>


Simulation::Simulation(SimParam p) : P_(std::move(p)) {}

void Simulation::init(int num_of_p) {
    particles_.resize(P_.N_max);
    if (num_of_p > P_.N_max) { std::cout << "Error: num_of_p = " << num_of_p << " with maximum number " << P_.N_max << std::endl; }
    assert(num_of_p > 0 && num_of_p <= P_.N_max);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> uni(0.f, 1.f);

    for (int i = 0; i < particles_.N_cur; ++i)
    {
        particles_.pos[i].x = uni(rng);
        particles_.pos[i].y = uni(rng);
        //Note that this should also work if float2 is manually defined
        particles_.omega[i] = (uni(rng) < 0.5f ? 1.f : -1.f);

        #if defined(USE_CUDA) && defined(__CUDACC__)
        build_subsets_cuda();

        #else
        build_subsets_cpu();

        #endif
    }
}
