#include "device_vector_alias.hpp"
#include "utilities.hpp"
#include <cmath>
#include <random>
#include "vortex_particle_mc.hpp"
#include "monte_carlo_bs.hpp"
#include <algorithm>

float sample_kernel_radius(std::mt19937& gen,
                                  const float* icdf, int L, float h)
{
    std::uniform_real_distribution<float> U(0.f, 1.f);
    float u = U(gen);

    float  fidx  = u * (L - 1);
    int    idx0  = int(fidx);
    int    idx1  = std::min(idx0 + 1, L - 1);
    float  t     = fidx - idx0;

    float  r_hat = icdf[idx0] * (1.f - t) + icdf[idx1] * t;
    return r_hat * h;
}
int sample_vorticity_idx_cdf(std::mt19937& gen,
                                    const float* cdf, int N)
{
    std::uniform_real_distribution<float> U(0.f, 1.f);
    float u = U(gen);

    // Use binary search to find the particle
    auto it = std::upper_bound(cdf, cdf + N, u);
    int  j  = int(it - cdf);
    return (j == N ? N-1 : j);
}

float2 sample_disk_biot_savart(std::mt19937& gen,
                                      const float2& center, float R)
{
    std::uniform_real_distribution<float> U(0.f, 1.f);

    float phi = 2.f * float(M_PI) * U(gen);
    float u   = U(gen);
    float r   = R * std::sqrt(u);             // p(r) ∝ 1/r → CDF ∝ r²
    return {center.x + r * std::cos(phi),
            center.y + r * std::sin(phi)};
}

