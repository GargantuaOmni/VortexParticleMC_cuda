#include "vortex_particle_mc.hpp"        // struct&class
#include "spatial_hasher.hpp"            // SpatialHasher
#include <random>
#include <numeric>                       // accumulate
#include <cuda_runtime.h>

/* ---------------- 1) 申请粒子内存 ---------------- */
void Simulation::init()
{
    // ① 粒子上限 & 当前粒子数
    particles_.N_max = P_.N_max;
    particles_.N_cur = P_.N_max;

    particles_.pos.resize(P_.N_max);
    particles_.omega.resize(P_.N_max);

    // Only used for uniformly distributed examples
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> uni(0.f, 1.f);

    for(int i = 0; i < P_.N_max; ++i){
        particles_.pos[i] = {uni(rng), uni(rng)};
        particles_.omega[i] = (uni(rng) < 0.5f ? 1.f : -1.f);   //
    }


    sub_pos_.clear();  sub_neg_.clear();
    for(int i = 0; i < P_.N_max; ++i){
        (particles_.omega[i] > 0 ? sub_pos_ : sub_neg_).push_back(i);
    }

    /*  This part should be discussed, because it is about how coarse/fine of the spatial hashing cells.   */
    int rsh = P_.ResolutionX / int(P_.hwr) / 4;


    SpatialHasher hash_pos(rsh, rsh, Backend::CUDA);   // ← 关键：用 CUDA 后端
    hash_pos.resize(int(sub_pos_.size()));             // 把 vp_cell / sort 拉到合适大小

    /* ------------------------------------------------
     * 4) 把粒子坐标拷上 GPU，调用 build()
     * ------------------------------------------------ */
    // 4-1 先准备 device 缓冲 (一次性，可改成成员变量持久化)
    float *d_px, *d_py;
    cudaMalloc(&d_px, P_.N_max * sizeof(float));
    cudaMalloc(&d_py, P_.N_max * sizeof(float));

    // 拆 X / Y → host 临时 buffer
    static std::vector<float> px, py;
    px.resize(P_.N_max);  py.resize(P_.N_max);
    for(int i=0;i<P_.N_max;++i){
        px[i] = particles_.pos[i].x;
        py[i] = particles_.pos[i].y;
    }

    // 拷前 N_cur 粒子
    cudaMemcpy(d_px, px.data(), particles_.N_cur*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_py, py.data(), particles_.N_cur*sizeof(float), cudaMemcpyHostToDevice);

    // 4-2 device 版 sub_index
    int *d_sub_pos;
    cudaMalloc(&d_sub_pos, sub_pos_.size()*sizeof(int));
    cudaMemcpy(d_sub_pos, sub_pos_.data(), sub_pos_.size()*sizeof(int), cudaMemcpyHostToDevice);

    // 4-3 调 hasher.build —— CUDA 分支
    hash_pos.build(particles_.pos,      //
                   sub_pos_,            //
                   P_.Lx, P_.Ly);       //

    // 4-4 可选：把 cell_num 拷回检查
    std::vector<int> cell_num_cpu(hash_pos.data().cell_num.size());
    cudaMemcpy(cell_num_cpu.data(),
               hash_pos.data().cell_num.data(),    // device ptr
               cell_num_cpu.size()*sizeof(int),
               cudaMemcpyDeviceToHost);

    int sum = std::accumulate(cell_num_cpu.begin(), cell_num_cpu.end(), 0);
    assert(sum == int(sub_pos_.size()));           //
}
