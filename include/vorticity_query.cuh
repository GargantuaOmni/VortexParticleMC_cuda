#pragma once
#include "vortex_particle_mc.hpp"   // VorticityView, float2
#include "device_vector_alias.hpp"

inline __device__
float queryVorticityAbs(const VorticityView& v,
                        float2 p, int sign,        // ±1
                        bool use_signed, bool exclude_center,
                        float h_w, bool periodic)
{
    float sum = 0.f;
    if(periodic){
        p.x = p.x - floorf(p.x / v.Lx) * v.Lx;
        p.y = p.y - floorf(p.y / v.Ly) * v.Ly;
    }else{
        if(p.x<0.f||p.x>v.Lx||p.y<0.f||p.y>v.Ly) return 0.f;
    }
    int cx = min(int(p.x / v.Lx * v.rshx), v.rshx-1);
    int cy = min(int(p.y / v.Ly * v.rshy), v.rshy-1);

    const float hh = h_w * h_w;

    for(int ox=-1; ox<=1; ++ox)
        for(int oy=-1; oy<=1; ++oy)
        {
            int gx = cx + ox, gy = cy + oy;
            if(periodic){
                gx = (gx % v.rshx + v.rshx) % v.rshx;
                gy = (gy % v.rshy + v.rshy) % v.rshy;
            }else{
                if(gx<0||gy<0||gx>=v.rshx||gy>=v.rshy) continue;
            }

            int c   = gx * v.rshy + gy;
            int beg = v.cell_acc[c];
            int end = (c+1 < v.rshx*v.rshy) ? v.cell_acc[c+1] : v.cnt_view;

            for(int k = beg; k < end; ++k){
                int local = v.vp_sort[k];
                int I     = v.sub[local];

                float2 d = { v.pos[I].x - p.x, v.pos[I].y - p.y };
                // TODO: A lot of branches here, we need to optimize it
                if(periodic){
                    if(d.x >  v.Lx - 2*v.Lx / v.rshx) d.x -= v.Lx;
                    if(d.x < -v.Lx + 2*v.Lx / v.rshx) d.x += v.Lx;
                    if(d.y >  v.Ly - 2*v.Ly / v.rshy) d.y -= v.Ly;
                    if(d.y < -v.Ly + 2*v.Ly / v.rshy) d.y += v.Ly;
                }
                float r2 = d.x*d.x + d.y*d.y;
                if(r2 > hh)                 continue;
                if(exclude_center && r2<1e-10f) continue;
                if(sign>0 && v.w[I]<=0)     continue;
                if(sign<0 && v.w[I]>=0)     continue;

                float wval = use_signed ? v.w[I] : fabsf(v.w[I]);
                float r    = sqrtf(r2);
                sum += wval * cubicSplinePDF(r, h_w);
            }
        }
    return sum;
}


inline __device__
float queryVorticityAbsNaive(const VorticityView& v,
                  float2  p,
                  int     sign,
                  bool    use_signed,
                  bool    exclude_center,
                  float   h_w)
{
    float sum = 0.f;
    for (int i = 0; i < v.N; ++i){
        float dx = p.x - v.pos[i].x;
        float dy = p.y - v.pos[i].y;
        float r = hypotf(dx, dy);   // 防 0
        if (exclude_center && r < 1.e-5f) continue;
        if(sign>0 && v.w[i]<=0) continue;
        if(sign<0 && v.w[i]>=0) continue;

        float wval = use_signed ? v.w[i] : fabsf(v.w[i]);
        sum += wval * cubicSplinePDF(r, h_w);
    }
    return sum;
}