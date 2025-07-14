#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/count.h>
#include <vortex_particle_mc.hpp>
#include <iostream>


void Simulation::build_subsets_cuda()
{
    const int N = particles_.N_cur;

    std::cout << "Debug -1"  << std::endl;
    std::cout << "particles_.N_cur   "  << particles_.N_cur << std::endl;
    std::cout << "particles_.omega.size()    "  << particles_.omega.size() << std::endl;
    cnt_pos_ = thrust::count_if(particles_.omega.begin(),
                                particles_.omega.end(),
                                [] __device__ (float w){ return w > 0.f; });

    cnt_neg_ = thrust::count_if(particles_.omega.begin(),
                                particles_.omega.end(),
                                [] __device__ (float w){ return w < 0.f; });

    std::cout << "Debug -0.5"  << std::endl;
    sub_pos_.resize(cnt_pos_);
    sub_neg_.resize(cnt_neg_);
    cdf_pos_.resize(cnt_pos_);
    cdf_neg_.resize(cnt_neg_);

    std::cout << "Debug 0"  << std::endl;

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

    std::cout << "Debug 1"  << std::endl;
    dvec<float> tmp(cnt_pos_);
    if (cnt_pos_ == 0) cum_pos_ = 0.0;
    else {
        thrust::copy_if(particles_.omega.begin(), particles_.omega.begin()+N,
                        tmp.begin(),
                        [] __device__ (float w){ return w > 0.f; });
        thrust::inclusive_scan(tmp.begin(), tmp.end(), cdf_pos_.begin());
        cum_pos_ = cdf_pos_.back();
    }

    if (cum_pos_ > 0.0f) {
        float inv = 1.0f / cum_pos_;
        thrust::transform(cdf_pos_.begin(), cdf_pos_.end(),
                          cdf_pos_.begin(),
                          [inv] __device__ (float x){ return x * inv; });
    }

    std::cout << "Debug 2"  << std::endl;

    tmp.resize(cnt_neg_);
    if (cnt_neg_ == 0) cum_neg_ = 0.0;
    else {
        thrust::copy_if(particles_.omega.begin(), particles_.omega.begin()+N,
                    tmp.begin(),
                    [] __device__ (float w){ return w < 0.f; });
        thrust::transform(tmp.begin(), tmp.end(), tmp.begin(), thrust::negate<float>()); // Abs
        thrust::inclusive_scan(tmp.begin(), tmp.end(), cdf_neg_.begin());
        cum_neg_ = cdf_neg_.back();
    }

    std::cout << "Debug 3"  << std::endl;

    if (cum_neg_ > 0.0f) {
        float inv = 1.0f / cum_neg_;
        thrust::transform(cdf_neg_.begin(), cdf_neg_.end(),
                          cdf_neg_.begin(),
                          [inv] __device__ (float x){ return x * inv; });
    }

    std::cout << "Using Thrust to build particle sets"  << std::endl;
}
