#pragma once

#include "device_vector_alias.hpp"

#define M_PI 3.14159265358979323846

inline __device__ float cubicSplinePDF(float r, float h_w)
{
    const float pi   = 3.141592653589793f;
    const float sig2 = 40.f / (7.f * pi * h_w * h_w);
    float q = r / h_w;
    if (q >= 0.f && q <= 0.5f)
        return sig2 * (6.f * (q*q*q - q*q) + 1.f);
    if (q > 0.5f && q < 1.f)
        return sig2 * 2.f * (1.f - q) * (1.f - q) * (1.f - q);
    return 0.f;
}


void build_cubic_rcdf(float h_w, int length, dvec<float>& cdf);

void cdf_to_icdf(const dvec<float>& cdf, float h_w, dvec<float>& icdf);

