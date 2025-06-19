#include "spatial_hash.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

void FillCells_cpu(int rshx,int rshy,float Lx,float Ly,int N,
                   std::span<const int>   sub,
                   std::span<const float> px,std::span<const float> py,
                   std::span<int> cell_num,std::span<int> cell_acc,
                   std::span<int> vp_cell)
{

    std::fill(cell_num.begin(), cell_num.end(), 0);

    const float inv_hx = rshx / Lx;
    const float inv_hy = rshy / Ly;

    for (int i = 0; i < N; ++i) {      // ← Only to N_current
        int j  = sub[i];
        int ix = int(std::floor(px[j] * inv_hx));
        int iy = int(std::floor(py[j] * inv_hy));
        ix = std::clamp(ix, 0, rshx - 1);
        iy = std::clamp(iy, 0, rshy - 1);

        int cell = ix * rshy + iy;
        vp_cell[i] = cell;
        ++cell_num[cell];
    }
    // std::exclusive_scan(cell_num.begin(), cell_num.end(), cell_acc.begin(), 0);
    int cell_cnt = rshx * rshy;
    cell_acc[0] = 0;
    for(int c = 1; c < cell_cnt; ++c)
        cell_acc[c] = cell_acc[c-1] + cell_num[c-1];   // exclusive
}

void CountingSort_cpu(int N,
                      std::span<int> cell_num,
                      std::span<const int> cell_acc,
                      std::span<const int> vp_cell,
                      std::span<int> vp_map)
{
    for (int i = N - 1; i >= 0; --i) {
        int c   = vp_cell[i];                    //
        int dst = --cell_num[c] + cell_acc[c];   // exclusive
        vp_map[dst] = i;                            //

    }
}