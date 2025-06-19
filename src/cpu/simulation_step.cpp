#include "device_vector_alias.hpp"
#include "vortex_particle_mc.hpp"
#include "monte_carlo_bs.hpp"
#include <random>


void Simulation::step_cpu()
{
    particles_.pos_temp.resize(particles_.N_cur);
    std::copy(particles_.pos.begin(), particles_.pos.begin() + particles_.N_cur, particles_.pos_temp.begin());

    std::mt19937 rng(123);
    MCBSCtx ctx;
    ctx.sub_pos   = &sub_pos_;
    ctx.sub_neg   = &sub_neg_;
    ctx.cdf_pos   = &cdf_pos_;
    ctx.cdf_neg   = &cdf_neg_;
    ctx.icdf_kernel = &icdf_kernel;      //
    ctx.kernel_L    = 0;
    ctx.h_w         = P_.h_w;      //
    ctx.periodic    = false;
    ctx.Lx = P_.Lx;  ctx.Ly = P_.Ly;
    ctx.cum_pos     = cum_pos_;
    ctx.cum_neg     = cum_neg_;
    ctx.pos_view    = makePosView();
    ctx.neg_view    = makeNegView();

    for(int i = 0; i < particles_.N_cur; ++i)
    {
        float2 dst      = particles_.pos[i];
        float2 dst_left = particles_.pos_left[i];
        float2 dst_right= particles_.pos_right[i];
        float2 dst_top  = particles_.pos_top[i];
        float2 dst_bot  = particles_.pos_bottom[i];

        auto vpos = monte_carlo_bs_cpu(
                        dst, +1, /*aux=*/1,
                        dst_left, dst_right, dst_top, dst_bot,
                        particles_, ctx, P_.N1, P_.N2, rng);

        auto vneg = monte_carlo_bs_cpu(
                        dst, -1, /*aux=*/1,
                        dst_left, dst_right, dst_top, dst_bot,
                        particles_, ctx, P_.N1, P_.N2, rng);

        float2 u       =  vpos.main   - vneg.main;
        float2 u_left  =  vpos.left   - vneg.left;
        float2 u_right =  vpos.right  - vneg.right;
        float2 u_top   =  vpos.top    - vneg.top;
        float2 u_bot   =  vpos.bottom - vneg.bottom;

        particles_.pos_temp[i]    += u      * P_.dt;
        particles_.pos_left[i]   += u_left  * P_.dt;
        particles_.pos_right[i]  += u_right * P_.dt;
        particles_.pos_top[i]    += u_top   * P_.dt;
        particles_.pos_bottom[i] += u_bot   * P_.dt;
    }

    std::copy(particles_.pos_temp.begin(), particles_.pos_temp.begin() + particles_.N_cur, particles_.pos.begin());

    if (ctx.periodic) {
        for(int i = 0; i < particles_.N_cur; ++i){
            auto wrap = [this](float& v, float L){
                v = v - std::floor(v / L) * L;  //
            };
            wrap(particles_.pos[i].x, P_.Lx);     wrap(particles_.pos[i].y, P_.Ly);
            wrap(particles_.pos_left[i].x,   P_.Lx); wrap(particles_.pos_left[i].y,   P_.Ly);
            wrap(particles_.pos_right[i].x,  P_.Lx); wrap(particles_.pos_right[i].y,  P_.Ly);
            wrap(particles_.pos_top[i].x,    P_.Lx); wrap(particles_.pos_top[i].y,    P_.Ly);
            wrap(particles_.pos_bottom[i].x, P_.Lx); wrap(particles_.pos_bottom[i].y, P_.Ly);
        }
    }

    std::cout << "step finished\n";
}