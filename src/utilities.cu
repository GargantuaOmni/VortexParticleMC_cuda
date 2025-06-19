//src/utilities
#include <device_vector_alias.hpp>
#include <utilities.hpp>
#include <thrust/host_vector.h>
#if defined(USE_CUDA) && defined(__CUDACC__)
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cpp/memory.h>   // CPU fallback
#endif

void build_cubic_rcdf(float h_w, int length,
                      dvec<float>& cdf)            // length
{
    cdf.resize(length);
    const float h_a = h_w / length;                //

    /* ------ thurst transform to make r·pdf(r) ------ */
#if defined(USE_CUDA) && defined(__CUDACC__)
    thrust::transform(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(length),
        cdf.begin(),
        [=] __device__ (int i)
        {
            float x = (0.5f + i) * h_a;            //
            return cubicSplinePDF(x, h_w) * x;     // r·pdf(r)
        });
    /* CDF without normalization */
    thrust::inclusive_scan(cdf.begin(), cdf.end(), cdf.begin());
#else
    float acc = 0.f;
    for (int i=0;i<length;++i){
        float x = (0.5f + i) * h_a;
        acc += cubicSplinePDF(x, h_w) * x;
        cdf[i] = acc;
    }
#endif

    /* Normalization */
    float total = cdf.back();
#if defined(USE_CUDA) && defined(__CUDACC__)
    float inv = 1.f / total;
    thrust::transform(cdf.begin(), cdf.end(), cdf.begin(),
                      [inv] __device__ (float v){ return v * inv; });
#else
    for(float& v:cdf) v /= total;
#endif
}

void cdf_to_icdf(const dvec<float>& cdf, float h_w,
                 dvec<float>& icdf)
{
    const int L  = static_cast<int>(cdf.size());
    const float dh = 1.f / L;
    icdf.resize(L);

#if defined(USE_CUDA) && defined(__CUDACC__)
    /* Using Host_vector to deal with CDF to ICDF */
    thrust::host_vector<float> cdf_h(cdf.begin(), cdf.end());
    thrust::host_vector<float> icdf_h(L);
#else
    const std::vector<float>& cdf_h = cdf;   // alias
    std::vector<float>        icdf_h(L);
#endif

    /*  */
    auto lookup = [&cdf_h,L](float u)
    {
        int ptr = 0;
        while (ptr < L && u > cdf_h[ptr]) ++ptr;
        if (ptr == 0){
            float λ = (cdf_h[0]-u)/cdf_h[0];
            return (1-λ)*0.5f;
        }
        if (ptr == L){
            float λ = (1-u)/(1-cdf_h[L-1]);
            return (1-λ)*(L-0.5f);
        }
        float λ = (cdf_h[ptr]-u)/(cdf_h[ptr]-cdf_h[ptr-1]);
        return (1-λ)*(ptr+0.5f) + λ*(ptr-0.5f);
    };

    for(int i = 0; i < L; ++i){
        float u   = (i + 0.5f) * dh;
        float idx = lookup(u);          // 0.5‒(L-0.5)
        icdf_h[i] = idx * dh * h_w;     //
    }

#if defined(USE_CUDA) && defined(__CUDACC__)
    /* Copy back */
    thrust::copy(icdf_h.begin(), icdf_h.end(), icdf.begin());
#else
    icdf = icdf_h;                      // std::vector 直接赋值
#endif
}


#ifdef __CUDA_ARCH__
#define GPU_ASSERT(cond)  if(!(cond)){ asm("trap;"); }
#else
#define GPU_ASSERT(cond)  assert(cond)
#endif


#if defined(USE_CUDA) && defined(__CUDACC__)
template<typename T>
void assert_equal(const thrust::device_vector<T>& d,
                  const std::vector<T>&           h,
                  const char* name)
{
    std::vector<T> tmp(d.size());
    cudaMemcpy(tmp.data(), thrust::raw_pointer_cast(d.data()),
               d.size()*sizeof(T), cudaMemcpyDeviceToHost);

    for(size_t i=0;i<tmp.size();++i)
        if(tmp[i] != h[i]){
            std::cerr<<name<<" mismatch at "<<i
                     <<" gpu="<<tmp[i]<<" cpu="<<h[i]<<'\n';
            assert(false);
        }
}
#endif
