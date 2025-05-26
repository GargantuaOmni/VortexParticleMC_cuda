//
// Created by Gargantua on 5/20/2025.
//
#include "spatial_hash.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <vector>
#include <omp.h>
#include <atomic>

void FillCells_omp(int rshx,int rshy,
                   float Lx,float Ly,
                   int N,
                   std::span<const int>   sub,
                   std::span<const float> px,
                   std::span<const float> py,
                   std::span<int> cell_num,
                   std::span<int> cell_acc,
                   std::span<int> vp_cell)
{
    const int cell_cnt = rshx * rshy;
    std::fill(cell_num.begin(), cell_num.end(), 0);

    const float inv_hx = rshx / Lx;
    const float inv_hy = rshy / Ly;

#pragma omp parallel
    {
        static thread_local std::vector<int> local;
        if (local.size() != cell_cnt) local.resize(cell_cnt);
        std::fill(local.begin(), local.end(), 0);

#pragma omp for schedule(static)
        for (int i = 0; i < N; ++i) {
            int j  = sub[i];
            int ix = int(std::floor(px[j] * inv_hx));
            int iy = int(std::floor(py[j] * inv_hy));
            ix = std::clamp(ix, 0, rshx - 1);
            iy = std::clamp(iy, 0, rshy - 1);

            int cell = ix * rshy + iy;
            vp_cell[i] = cell;
            ++local[cell];
        }

#pragma omp critical
        for (int c = 0; c < cell_cnt; ++c)
            cell_num[c] += local[c];
    }

    std::exclusive_scan(cell_num.begin(), cell_num.end(),
                        cell_acc.begin(), 0);
}

void CountingSort_omp(int N,
                      std::span<int>       cell_num,
                      std::span<const int> cell_acc,
                      std::span<const int> vp_cell,
                      std::span<int>       vp_map)
{
    #pragma omp parallel for schedule(static)
    for (int i = N - 1; i >= 0; --i) {
        int c = vp_cell[i];

        int old = std::atomic_ref<int>(cell_num[c]).fetch_sub(1, std::memory_order_relaxed);
        int dst = old - 1 + cell_acc[c];
        vp_map[dst] = i;
    }
}