//
// Created by Gargantua on 9/25/2025.
//
// naive_biot_savart.cu
#include <cuda_runtime.h>
#include "device_vector_alias.hpp"   //
#include "utilities.hpp"
#include "monte_carlo_bs.cuh"

#ifndef HD
#define HD __host__ __device__
#endif


__global__ void biot_savart_naive_kernel(
    const float2* __restrict__ pos,
    const float*  __restrict__ gamma,  //
    int N, float2* __restrict__ out,
    float Lx, float Ly, bool periodic,
    float eps)                          //
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float2 xi = pos[i];
    float2 ui = make_float2(0.f, 0.f);
    float  eps2 = eps * eps;

    for (int j = 0; j < N; ++j){
        if (j == i) continue;
        float2 d;
        d.x = wrap_min_image(xi.x - pos[j].x, Lx, periodic);
        d.y = wrap_min_image(xi.y - pos[j].y, Ly, periodic);

        float2 kij = bs_kernel_2d(d, eps2);
        ui.x += gamma[j] * kij.x;
        ui.y += gamma[j] * kij.y;
    }
    out[i] = ui;
}
