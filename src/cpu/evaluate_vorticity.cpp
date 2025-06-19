#include "device_vector_alias.hpp"
#include "vortex_particle_mc.hpp"
#include <cmath>
#include "utilities.hpp"
#include <iostream>
#include "vector"

float evalVorticityAbs_cpu(const VorticityView& v,
                           const float2& p,
                           int sign,                   // +1 = positive, -1 = negative
                           bool use_signed     = false,
                           bool exclude_center = false,
                           float h_w           = 0.04f)   // Support radius
{
    if(p.x<0.f||p.x>1.f||p.y<0.f||p.y>1.f) return 0.f;
    float sum = 0.f;
    auto visit = [&](int I){
        float2 d = {v.pos[I].x - p.x, v.pos[I].y - p.y};
        float r  = std::sqrt(d.x*d.x + d.y*d.y);
        if(exclude_center && r < 1e-5f) return;
        if(r > h_w)                     return;

        float wval = use_signed ? v.w[I] : std::fabs(v.w[I]);
        if(sign>0 && v.w[I]<=0) return;
        if(sign<0 && v.w[I]>=0) return;
        sum += wval * cubicSplinePDF(r, h_w);
    };

    if(v.vp_sort && v.cell_acc){                 // Hash
        int cx = std::min(int(p.x * v.rshx / v.Lx), v.rshx - 1);   //
        int cy = std::min(int(p.y * v.rshy / v.Ly), v.rshy - 1);

        for(int ox=-1; ox<=1; ++ox)
            for(int oy=-1; oy<=1; ++oy){
                int gx=cx+ox, gy=cy+oy;
                if(gx<0||gy<0||gx>=v.rshx||gy>=v.rshy) continue;
                //// TODO: change this part to periodic boundary conditions
                int c   = gx*v.rshy+gy;
                int beg = v.cell_acc[c];
                int end = (c+1<v.rshx * v.rshy) ? v.cell_acc[c+1] : v.cnt_view;
                // if (cx == v.rshx-1) std::cout << "beg = " << beg << " end = " << end << std::endl;
                for(int k=beg;k<end;++k){
                    int local = v.vp_sort[k];
                    int I     = v.sub[local];     // Local -> Global
                    visit(I);
                }
            }
    }else{                                       /* Naïve (without hashing) */
        for(int I=0; I<v.N; ++I) visit(I);
    }
    return sum;
}