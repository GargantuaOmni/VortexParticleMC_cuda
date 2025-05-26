#include <iostream>
#include <chrono>
#include <cstdlib>
#include <vector>
#include <random>
#include <string>
#include "spatial_hash.hpp"      // 纯函数声明

#include <numeric>   // accumulate
#include <cassert>
#include <omp.h>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

enum class Backend { CPU, OMP, CUDA };
Backend parse_backend(int argc, char* argv[])
{
    for (int i = 1; i < argc; ++i) {
        std::string_view arg{argv[i]};
        if (arg.rfind("--backend=", 0) == 0) {        // 前缀匹配
            std::string_view val = arg.substr(10);
            if (val == "cpu")  return Backend::CPU;
            if (val == "omp")  return Backend::OMP;
            if (val == "cuda") return Backend::CUDA;
            std::cerr << "Unknown backend: " << val << '\n';
            std::exit(EXIT_FAILURE);
        }
    }
    return Backend::CPU;                              // 默认
}

int main(int argc, char* argv[]){
    /* -------------- 常量与上限 ---------------- */
    Backend backend = parse_backend(argc, argv);
    std::cout << "[Backend] " << (backend==Backend::CPU ? "CPU" :
                                   backend==Backend::OMP ? "OMP" : "CUDA")
              << '\n';

    const int   N_max = 50'000;            // 绝不会超过此值
    const int   rshx = 128, rshy = 128;
    const float Lx   = 1.0f, Ly = 1.0f;
    const int   cell_cnt = rshx * rshy;

    /* -------------- 一次性预分配 -------------- */
    std::vector<float> pos_x(N_max), pos_y(N_max);
    std::vector<int>   vp_cell(N_max), vp_map(N_max), cell_num_copy(N_max);
    std::vector<int>   cell_num(rshx*rshy), cell_acc(rshx*rshy);
    std::vector<int>   sub_index(N_max);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.f,1.f);

    thrust::device_vector<float> d_px(N_max), d_py(N_max);
    thrust::device_vector<int>   d_cell_num(rshx*rshy), d_cell_acc(rshx*rshy);
    thrust::device_vector<int>   d_vp_cell(N_max), d_vp_map(N_max);
    thrust::device_vector<int>   d_sub_index(N_max);

    if (backend == Backend::OMP){ std::cout << "omp_get_max_threads() = " << omp_get_max_threads() << '\n'; }

    /* -------------- 时间循环 ------------------ */
    auto tic = std::chrono::high_resolution_clock::now();

    for(int frame = 0; frame < 1000; ++frame){
        /* 动态决定当前粒子数 */
        int N_current = 10'000 + (frame % 2000); // 举例：波动
        // 若 N_current 超过 N_max，需要扩容（见附注）
        assert(N_current <= pos_x.size());
        /* 随机更新前 N_current 粒子的位置 */
        for(int i=0;i<N_current;++i){
            pos_x[i] = dist(rng)*Lx;
            pos_y[i] = dist(rng)*Ly;
        }

        for (int i=0;i<N_current;++i) {
            sub_index[i] = i;
        }

        if (backend == Backend::OMP) {
            FillCells_omp(rshx,rshy,Lx,Ly,N_current, sub_index,
                  {pos_x.data(), static_cast<std::size_t>(N_current)},
                  {pos_y.data(), static_cast<std::size_t>(N_current)},
                  cell_num, cell_acc,
                  {vp_cell.data(), static_cast<std::size_t>(N_current)});
            auto cell_num_copy = cell_num;
            CountingSort_omp(N_current,
                             cell_num, cell_acc,
                             {vp_cell.data(), static_cast<std::size_t>(N_current)},
                             {vp_map.data(), static_cast<std::size_t>(N_current)});
        }

        else if (backend == Backend::CUDA) {
            cudaMemcpy(thrust::raw_pointer_cast(d_px.data()),
           pos_x.data(), N_current * sizeof(float),
           cudaMemcpyHostToDevice);

            cudaMemcpy(thrust::raw_pointer_cast(d_py.data()),
                       pos_y.data(), N_current * sizeof(float),
                       cudaMemcpyHostToDevice);    // We only copy effective data (particles)

            cudaMemcpy(thrust::raw_pointer_cast(d_px.data()),
           pos_x.data(), N_current * sizeof(float),
           cudaMemcpyHostToDevice);

            cudaMemcpy(thrust::raw_pointer_cast(d_sub_index.data()),
           sub_index.data(), N_current * sizeof(int),
           cudaMemcpyHostToDevice);

            FillCells_cuda(rshx, rshy, Lx, Ly, N_current,
                thrust::raw_pointer_cast(d_sub_index.data()),
               thrust::raw_pointer_cast(d_px.data()),
               thrust::raw_pointer_cast(d_py.data()),
               thrust::raw_pointer_cast(d_cell_num.data()),
               thrust::raw_pointer_cast(d_cell_acc.data()),
               thrust::raw_pointer_cast(d_vp_cell.data()));

            cudaMemcpy(cell_num.data(), thrust::raw_pointer_cast(d_cell_num.data()),
           cell_cnt * sizeof(int), cudaMemcpyDeviceToHost);
            auto cell_num_copy = cell_num;

            CountingSort_cuda(N_current,
                              thrust::raw_pointer_cast(d_cell_num.data()),
                              thrust::raw_pointer_cast(d_cell_acc.data()),
                              thrust::raw_pointer_cast(d_vp_cell.data()),
                              thrust::raw_pointer_cast(d_vp_map.data()));

            cudaMemcpy(vp_map.data(), thrust::raw_pointer_cast(d_vp_map.data()),
           N_current * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(cell_num.data(), thrust::raw_pointer_cast(d_cell_num.data()),
           cell_cnt * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(vp_cell.data(), thrust::raw_pointer_cast(d_vp_cell.data()),
                       N_current * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(cell_acc.data(), thrust::raw_pointer_cast(d_cell_acc.data()),
                       cell_cnt * sizeof(int), cudaMemcpyDeviceToHost);
        }

        else {
            FillCells_cpu(rshx,rshy,Lx,Ly,N_current,sub_index,
                  {pos_x.data(), static_cast<std::size_t>(N_current)},
                  {pos_y.data(), static_cast<std::size_t>(N_current)},
                  cell_num, cell_acc,
                  {vp_cell.data(), static_cast<std::size_t>(N_current)});
            auto cell_num_copy = cell_num;
            CountingSort_cpu(N_current,
                             cell_num, cell_acc,
                             {vp_cell.data(), static_cast<std::size_t>(N_current)},
                             {vp_map.data(), static_cast<std::size_t>(N_current)});
        }

        /* 调试断言 */
        int sum = std::accumulate(cell_num.begin(), cell_num.end(), 0);
        assert(sum==0);                       // 计数已被递减完
        // 也可备份 cell_num 在排序前验 sum==N_current

        for(int c = 0; c < rshx*rshy; ++c){
            for(int k = 0; k < cell_num_copy[c]; ++k){
                int p = vp_map[cell_acc[c] + k];
                assert(vp_cell[p] == c);
            }
        }
    }

    auto toc = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(toc-tic).count();
    std::string backend_name;
    if (backend == Backend::CUDA){ backend_name = "CUDA"; }
    else if (backend == Backend::OMP){ backend_name = "OMP"; }
    else if (backend == Backend::CPU){ backend_name = "CPU"; }
    else { backend_name = "Unknown"; }

    std::cout << backend_name  << "100 frames  cost = " << ms << " ms\n";
}