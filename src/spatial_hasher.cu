#include "spatial_hasher.hpp"
#include "spatial_hash.hpp"
#include <span>

SpatialHasher::SpatialHasher(int rx,int ry,Backend b)
: backend_(b)
{
    hash_.rshx = rx;
    hash_.rshy = ry;
    int cell_cnt = rx * ry;
    hash_.cell_num.resize(cell_cnt);
    hash_.cell_acc.resize(cell_cnt);
}

void SpatialHasher::resize(int N_max)
{
    hash_.vp_cell.resize(N_max);
    hash_.vp_sort.resize(N_max);
}

static void split_xy(const std::vector<glm::vec2>& pos,
                     std::vector<float>& px,
                     std::vector<float>& py)
{
    px.resize(pos.size());
    py.resize(pos.size());
    size_t i;
    for(i=0;i<pos.size();++i){
        px[i] = pos[i].x;
        py[i] = pos[i].y;
    }
}

void SpatialHasher::build(const std::vector<glm::vec2>& pos,
                          const std::vector<int>&       sub,
                          float Lx,float Ly)
{
    int N = static_cast<int>(sub.size());
    if(N > hash_.vp_cell.size()) resize(N);

    static thread_local std::vector<float> px, py;
    split_xy(pos, px, py);

    switch(backend_){
        case Backend::CPU:
            FillCells_cpu(hash_.rshx, hash_.rshy, Lx, Ly, N,
                          sub,
                          std::span<const float>{px}.first(N),
                          std::span<const float>{py}.first(N),
                          hash_.cell_num, hash_.cell_acc, hash_.vp_cell);


            CountingSort_omp(N,
                             hash_.cell_num, hash_.cell_acc,
                             std::span<const int>{hash_.vp_cell}.first(N),
                             std::span<int>{hash_.vp_sort}.first(N));
            break;
        case Backend::OMP:    // Still problematic
            FillCells_omp(hash_.rshx, hash_.rshy, Lx, Ly, N,
                          sub,
                          std::span<const float>{px}.first(N),
                          std::span<const float>{py}.first(N),
                          hash_.cell_num, hash_.cell_acc, hash_.vp_cell);
            break;
        case Backend::CUDA:
            // FillCells_cuda(... d_px, d_py ...)
            break;
    }
}
