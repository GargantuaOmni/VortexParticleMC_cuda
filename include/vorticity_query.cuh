#pragma once
#include "vortex_particle_mc.hpp"   // VorticityView, float2
#include "device_vector_alias.hpp"

inline __device__
float queryVorticityAbs(const VorticityView& v,
                        float2 p, int sign,        // ±1
                        bool use_signed, bool exclude_center,
                        float h_w)
{
    float sum = 0.f;
    if (p.x<0.f||p.x>1.f||p.y<0.f||p.y>1.f) return 0.f;

    /*  */
    int cx = min(int(p.x * v.rshx), v.rshx - 1);
    int cy = min(int(p.y * v.rshy), v.rshy - 1);

    /* */
    for(int ox=-1; ox<=1; ++ox)
        for(int oy=-1; oy<=1; ++oy)
        {
            int gx = cx+ox, gy = cy+oy;
            if(gx<0||gy<0||gx>=v.rshx||gy>=v.rshy) continue;
            // TODO: change it into periodic boundary

            int c   = gx*v.rshy + gy;
            int beg = v.cell_acc[c];
            int end = (c+1 < v.rshx * v.rshy) ? v.cell_acc[c+1] : v.cnt_view;

            /* visit particles */
            for(int k = beg; k < end; ++k){
                int local = v.vp_sort[k];
                int I     = v.sub[local];

                float2 d = { v.pos[I].x - p.x, v.pos[I].y - p.y };
                float  r = hypotf(d.x, d.y);
                if(r > h_w) continue;
                if(exclude_center && r < 1e-5f) continue;
                if(sign>0 && v.w[I]<=0) continue;
                if(sign<0 && v.w[I]>=0) continue;

                float wval = use_signed ? v.w[I] : fabsf(v.w[I]);
                sum += wval * cubicSplinePDF(r, h_w);      //
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