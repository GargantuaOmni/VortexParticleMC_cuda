#include <vortex_particle_mc.hpp>
#include <vector>
#include <device_vector_alias.hpp>
#include <iostream>

void Simulation::build_subsets_cpu()
{
    const int N = particles_.N_cur;
    sub_pos_.clear();   sub_neg_.clear();
    cdf_pos_.clear();   cdf_neg_.clear();

    float cum_p = 0.f, cum_n = 0.f;

    for(int i = 0;i < N;++i){
        float w = particles_.omega[i];
        if (w > 0){
            sub_pos_.push_back(i);
            cum_p += w;
            cdf_pos_.push_back(cum_p);
        } else if (w < 0){
            sub_neg_.push_back(i);
            cum_n -= w;            // Use absolute values
            cdf_neg_.push_back(cum_n);
        }
    }
    cnt_pos_ = int(sub_pos_.size());
    cnt_neg_ = int(sub_neg_.size());
    cum_pos_ = cum_p;
    cum_neg_ = cum_n;

    // Normalization
    if (cum_pos_ > 0.0f) {
        float inv = 1.0f / cum_pos_;
        for (float& v : cdf_pos_) v *= inv;
    }
    if (cum_neg_ > 0.0f) {
        float inv = 1.0f / cum_neg_;
        for (float& v : cdf_neg_) v *= inv;
    }

    std::cout << "Using CPU backend to build particle sets"  << std::endl;
}
