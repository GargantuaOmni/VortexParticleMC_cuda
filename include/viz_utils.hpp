#pragma once
#include <vector>
#include <string>
#include <filesystem>
#include <cmath>
#include <vortex_particle_mc.hpp>
#include "stb_image_write.h"

// -------- float2 / length --------
inline float length(float2 v){ return std::sqrt(v.x*v.x + v.y*v.y); }

/* ---------------------------------------------------------
   dump_vorticity_png
   - sim.eval_vorticity(x,y)  :
   - res                      :
   --------------------------------------------------------- */
void dump_vorticity_png(const Simulation& sim,
                        const std::string& file,
                        int res = 512)
{
    std::vector<unsigned char> img(res*res);
    float  max_abs = 0.f;

    for(int j=0;j<res;++j){
        float y = (j+0.5f)/res;
        for(int i=0;i<res;++i){
            float x = (i+0.5f)/res;
            float2 p = {x, y};
            float w = sim.eval_vorticity_cpu(p);
            max_abs = std::max(max_abs, std::fabs(w));
        }
    }
    float inv = (max_abs>0)? 127.f/max_abs : 1.f;

    for(int j=0;j<res;++j){
        float y = (j+0.5f)/res;
        for(int i=0;i<res;++i){
            float x = (i+0.5f)/res;
            float2 p = {x, y};
            float w = sim.eval_vorticity_cpu(p);
            int   idx= j*res+i;
            int   val= 128 + int(w*inv);
            img[idx] = static_cast<unsigned char>(std::clamp(val,0,255));
        }
    }
    stbi_write_png(file.c_str(), res, res, 1, img.data(), res);
}
