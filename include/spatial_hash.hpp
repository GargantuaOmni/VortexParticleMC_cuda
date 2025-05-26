// include/spatial_hash.hpp
#pragma once
#include <span>

// 只在参数里保留 "当前粒子数 N"。容量信息靠调用者保证不越界。
void FillCells_cpu(int rshx, int rshy,
                   float Lx, float Ly,
                   int   N_current,                  // ← 逻辑长度
                   std::span<const int>   sub,
                   std::span<const float> pos_x,     // size ≥ N_current
                   std::span<const float> pos_y,
                   std::span<int>  cell_num,         // size = rshx*rshy
                   std::span<int>  cell_acc,
                   std::span<int>  vp_cell);         // size ≥ N_current

void CountingSort_cpu(int N_current,
                      std::span<int>       cell_num,    // rshx*rshy
                      std::span<const int> cell_acc,
                      std::span<const int> vp_cell,
                      std::span<int>       vp_map);     // ≥ N_current



void FillCells_omp(int rshx, int rshy,
                   float Lx, float Ly,
                   int   N_current,                  // ← 逻辑长度
                   std::span<const int>   sub,
                   std::span<const float> pos_x,     // size ≥ N_current
                   std::span<const float> pos_y,
                   std::span<int>  cell_num,         // size = rshx*rshy
                   std::span<int>  cell_acc,
                   std::span<int>  vp_cell);         // size ≥ N_current

void CountingSort_omp(int N_current,
                      std::span<int>       cell_num,    // rshx*rshy
                      std::span<const int> cell_acc,
                      std::span<const int> vp_cell,
                      std::span<int>       vp_map);     // ≥ N_current


void FillCells_cuda(int rshx,int rshy,
                    float Lx,float Ly,
                    int   N,                      // N_current
                    const int* sub,
                    const float* pos_x,           // device ptr
                    const float* pos_y,
                    int*  cell_num,               // device ptr  rshx*rshy
                    int*  cell_acc,               // device ptr
                    int*  vp_cell);               // device ptr  N

void CountingSort_cuda(int  N,
                       int* cell_num,             // device ptr
                       const int* cell_acc,
                       const int* vp_cell,
                       int* vp_map);              // device ptr
