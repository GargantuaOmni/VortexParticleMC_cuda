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
                           float h_w           = 0.04f,
                           bool periodic       = false)   // Support radius
{
    float2 pos = {0.0f, 0.0f};
    if(periodic){
        float Lx = v.Lx, Ly = v.Ly;
        // wrap into [0,L)
        pos.x = p.x - std::floor(p.x / Lx) * Lx;
        pos.y = p.y - std::floor(p.y / Ly) * Ly;
    }else{
        if(p.x<0.f||p.x>v.Lx||p.y<0.f||p.y>v.Ly) return 0.f;
    }
    float sum = 0.f;
    auto visit = [&](int I){
        float2 d = {v.pos[I].x - pos.x, v.pos[I].y - pos.y};
        if(periodic){
            const float Lx = v.Lx, Ly = v.Ly;
            const float cellx = Lx / v.rshx;
            const float celly = Ly / v.rshy;

            if(d.x >  Lx - 2*cellx) d.x -= Lx;
            if(d.x < -Lx + 2*cellx) d.x += Lx;
            if(d.y >  Ly - 2*celly) d.y -= Ly;
            if(d.y < -Ly + 2*celly) d.y += Ly;
        }
        float r2 = d.x*d.x + d.y*d.y;
        if(r2 > h_w*h_w || (exclude_center && r2<1e-12f)) return;


        float wval = use_signed ? v.w[I] : std::fabs(v.w[I]);
        if(sign>0 && v.w[I]<=0) return;
        if(sign<0 && v.w[I]>=0) return;
        sum += wval * cubicSplinePDF(std::sqrt(r2), h_w);
    };

    if(v.vp_sort && v.cell_acc){                 // Hash
        int cx = std::min(int(pos.x * v.rshx / v.Lx), v.rshx - 1);   //
        int cy = std::min(int(pos.y * v.rshy / v.Ly), v.rshy - 1);

        for(int ox=-1; ox<=1; ++ox)
            for(int oy=-1; oy<=1; ++oy){
                int gx=cx+ox, gy=cy+oy;
                if(periodic){
                    gx = (gx % v.rshx + v.rshx) % v.rshx;   // wrap
                    gy = (gy % v.rshy + v.rshy) % v.rshy;
                }else{
                    if(gx<0||gy<0||gx>=v.rshx||gy>=v.rshy) continue;
                }
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