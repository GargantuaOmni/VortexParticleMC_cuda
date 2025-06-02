#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/count.h>
#include <vortex_particle_mc.hpp>

void Simulation::build_subsets_cuda()
{
    const int N = particles_.N_cur;

    cnt_pos_ = thrust::count_if(particles_.omega.begin(),
                                particles_.omega.begin()+N,
                                [] __device__ (float w){ return w > 0.f; });

    cnt_neg_ = thrust::count_if(particles_.omega.begin(),
                                particles_.omega.begin()+N,
                                [] __device__ (float w){ return w < 0.f; });

    sub_pos_.resize(cnt_pos_);
    sub_neg_.resize(cnt_neg_);
    cdf_pos_.resize(cnt_pos_);
    cdf_neg_.resize(cnt_neg_);

    auto idx_first = thrust::make_counting_iterator<int>(0);

    auto end_pos =
    thrust::copy_if(idx_first, idx_first+N,
                    particles_.omega.begin(),
                    sub_pos_.begin(),
                    [] __device__ (float w){ return w > 0.f; });

    auto end_neg =
    thrust::copy_if(idx_first, idx_first+N,
                    particles_.omega.begin(),
                    sub_neg_.begin(),
                    [] __device__ (float w){ return w < 0.f; });

    dvec<float> tmp(cnt_pos_);
    thrust::copy_if(particles_.omega.begin(), particles_.omega.begin()+N,
                    tmp.begin(),
                    [] __device__ (float w){ return w > 0.f; });
    thrust::inclusive_scan(tmp.begin(), tmp.end(), cdf_pos_.begin());
    cum_pos_ = cdf_pos_.back();

    tmp.resize(cnt_neg_);
    thrust::copy_if(particles_.omega.begin(), particles_.omega.begin()+N,
                    tmp.begin(),
                    [] __device__ (float w){ return w < 0.f; });
    thrust::transform(tmp.begin(), tmp.end(), tmp.begin(), thrust::negate<float>()); // Abs
    thrust::inclusive_scan(tmp.begin(), tmp.end(), cdf_neg_.begin());
    cum_neg_ = cdf_neg_.back();
}
