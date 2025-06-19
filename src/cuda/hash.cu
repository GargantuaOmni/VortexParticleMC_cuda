#include "spatial_hash.hpp"
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <cuda_runtime.h>

__global__ void FillCells_kernel(
        int rshx, int rshy, float inv_hx, float inv_hy,
        const int*   sub,        // 局部 → 全局   (N_sub)
        const float2* p,         // 全局粒子坐标 (N_total)
        int* cell_num, int* vp_cell, int N_sub)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N_sub) return;

    int I  = sub[i];             // 全局粒子索引
    float2 pos = p[I];           // ★ 直接取 float2
    int ix = int(floorf(pos.x * inv_hx));
    int iy = int(floorf(pos.y * inv_hy));
    ix = min(max(ix,0), rshx-1);
    iy = min(max(iy,0), rshy-1);

    int cell = ix * rshy + iy;   // 与 CPU 完全一致
    vp_cell[i] = cell;
    atomicAdd(cell_num + cell, 1);
}

__global__ void counting_sort_kernel(
    const int* vp_cell,int* cell_num,
    const int* cell_acc,int* vp_map,int N)
{
    for(int i = N-1-(blockIdx.x*blockDim.x+threadIdx.x);
        i >= 0;
        i -= blockDim.x*gridDim.x)
    {
        int c   = vp_cell[i];
        int dst = atomicSub(cell_num + c, 1) - 1 + cell_acc[c];
        vp_map[dst] = i;
    }
}

/* -------- Host wrapper -------- */
void FillCells_cuda(int rshx,int rshy,float Lx,float Ly,int N, const int* d_sub,
                    const float2* d_p,
                    int* d_cell_num,int* d_cell_acc,int* d_vp_cell)
{
    int cell_cnt = rshx * rshy;
    cudaMemset(d_cell_num, 0, cell_cnt * sizeof(int));

    float inv_hx = rshx / Lx;
    float inv_hy = rshy / Ly;

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    FillCells_kernel<<<blocks, threads>>>(rshx,rshy,inv_hx,inv_hy, d_sub,
                                           d_p,d_cell_num,d_vp_cell,N);
    cudaDeviceSynchronize();

    // 前缀和
    thrust::exclusive_scan(thrust::device,
                           d_cell_num, d_cell_num + cell_cnt,
                           d_cell_acc);
}

void CountingSort_cuda(int N,
                       int* d_cell_num,const int* d_cell_acc,
                       const int* d_vp_cell,int* d_vp_map)
{
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    counting_sort_kernel<<<blocks, threads>>>(d_vp_cell,
                                              d_cell_num,
                                              d_cell_acc,
                                              d_vp_map,
                                              N);
    cudaDeviceSynchronize();
}
