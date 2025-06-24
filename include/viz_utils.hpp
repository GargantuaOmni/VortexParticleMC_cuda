#pragma once
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <vortex_particle_mc.hpp>
#include "stb_image_write.h"
#include "device_vector_alias.hpp"

#if defined(USE_CUDA)
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cpp/memory.h>   // CPU fallback
#endif

// -------- float2 / length --------
inline float length(float2 v){ return std::sqrt(v.x*v.x + v.y*v.y); }

/* ---------------------------------------------------------
   dump_vorticity_png
   - sim.eval_vorticity(x,y)  :
   - res                      :
   --------------------------------------------------------- */
inline void dump_vorticity_png(const Simulation& sim,
                               const std::string& file,
                               int res = 512, bool periodic = false)
{
#if defined(USE_CUDA)
    thrust::host_vector<float2> posP(sim.particles_.pos);
    thrust::host_vector<float>  omgP(sim.particles_.omega);
    auto getPos = [&](int i){ return posP[i]; };
    auto getW   = [&](int i){ return omgP[i]; };
#else
    auto& pos = sim.particles_.pos;
    auto& omg = sim.particles_.omega;
    auto getPos = [&](int i){ return pos[i]; };
    auto getW   = [&](int i){ return omg[i]; };
#endif

    /* ---------- Lambda 评估 vorticity at p ---------- */
    const float h_w = sim.P_.h_w;
    auto vorticity_at = [&](float2 p)
    {
        float sum = 0.f;
        for(int i=0;i<sim.particles_.N_cur;++i){
            float2 d = {getPos(i).x - p.x, getPos(i).y - p.y};
            if(periodic){
                if(d.x > 0.5f) d.x -= 1.f; else if(d.x<-0.5f) d.x += 1.f;
                if(d.y > 0.5f) d.y -= 1.f; else if(d.y<-0.5f) d.y += 1.f;
            }
            float r2 = d.x*d.x + d.y*d.y;
            if(r2 > h_w*h_w || r2<1e-12f) continue;
            float w = std::fabs(getW(i));
            float r = std::sqrt(r2);
            sum += w * cubicSplinePDF(r,h_w);               //
        }
        return sum;
    };

    std::vector<unsigned char> img(res*res);
    float max_abs = 0.f;
    for(int j=0;j<res;++j){
        float y = (j+0.5f)/res;
        for(int i=0;i<res;++i){
            float x = (i+0.5f)/res;
            float w = vorticity_at({x,y});
            max_abs = std::max(max_abs,std::fabs(w));
        }
    }
    float inv = max_abs>0 ? 127.f/max_abs : 1.f;
    for(int j=0;j<res;++j){
        float y = (j+0.5f)/res;
        for(int i=0;i<res;++i){
            float x = (i+0.5f)/res;
            float w = vorticity_at({x,y});
            img[j*res+i] = (unsigned char)std::clamp(128+int(w*inv),0,255);
        }
    }
    stbi_write_png(file.c_str(),res,res,1,img.data(),res);
}

inline void dump_velocity_svg_centered(const Simulation& sim,
                                       const std::string& file,
                                       float vel_scale = 0.2f,
                                       bool periodic   = false)
{
    std::filesystem::create_directories(std::filesystem::path(file).parent_path());

#if defined(USE_CUDA)
    thrust::host_vector<float2> posH(sim.particles_.pos);
    thrust::host_vector<float2> velH(sim.particles_.vel);
    auto getPos=[&](int i){ return posH[i]; };
    auto getVel=[&](int i){ return velH[i]; };
#else
    auto& pos = sim.particles_.pos;
    auto& vel = sim.particles_.vel;
    auto getPos=[&](int i){ return pos[i]; };
    auto getVel=[&](int i){ return vel[i]; };
#endif

    std::ofstream ofs(file);
    if(!ofs.is_open()){ std::cerr<<"x open "<<file<<" failed\n"; return; }

    ofs << R"(<?xml version="1.0" standalone="no"?>)"
        << "\n<svg xmlns=\"http://www.w3.org/2000/svg\" "
        << "width=\"600\" height=\"600\" viewBox=\"0 0 1 1\">\n"
        << "<style>.arrow{stroke:#ff4d00;stroke-width:0.0015;"
        << "marker-end:url(#head);}</style>\n"
        << "<defs><marker id=\"head\" orient=\"auto\" "
        << "markerWidth=\"4\" markerHeight=\"4\" refX=\"0.1\" refY=\"2\">"
        << "<path d=\"M0,0 V4 L4,2 Z\" fill=\"#ff4d00\"/></marker></defs>\n";

    for(int i=0;i<sim.particles_.N_cur;++i){
        float2 p = getPos(i);
        float2 v = getVel(i);
        float len = length(v);
        if(len < 1e-8f) continue;
        float2 dir = {v.x/len, v.y/len};
        float half = 0.5f*vel_scale*len;
        float2 a = {p.x - half*dir.x, p.y - half*dir.y};
        float2 b = {p.x + half*dir.x, p.y + half*dir.y};
        if(periodic){
            a.x -= std::floor(a.x); a.y -= std::floor(a.y);
            b.x -= std::floor(b.x); b.y -= std::floor(b.y);
        }
        ofs << "<line class=\"arrow\" x1=\"" << a.x << "\" y1=\"" << 1-a.y
            << "\" x2=\"" << b.x << "\" y2=\"" << 1-b.y << "\"/>\n";
    }
    ofs << "</svg>\n";
}