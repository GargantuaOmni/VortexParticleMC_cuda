inline __host__ __device__
float cubicSplinePDF(float r, float h_w)
{
    const float pi   = 3.141592653589793f;
    const float sig2 = 40.f / (7.f * pi * h_w * h_w);
    float q = r / h_w;
    if (q >= 0.f && q <= 0.5f)
        return sig2 * (6.f * (q*q*q - q*q) + 1.f);
    if (q > 0.5f && q < 1.f)
        return sig2 * 2.f * (1.f - q) * (1.f - q) * (1.f - q);
    return 0.f;
}

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
                 dvec<float>& icdf)          //
{
    const int L = static_cast<int>(cdf.size());
    icdf.resize(L);
    const float dh = 1.f / L;

    /*  */
    auto lookup = [&cdf,L](float u)
    {
        int ptr = 0;
        while(ptr<L && u > cdf[ptr]) ++ptr;

        if(ptr==0){
            float λ = (cdf[0]-u)/cdf[0];
            return (1-λ)*0.5f;
        }
        if(ptr==L){
            float λ = (1-u)/(1-cdf[L-1]);
            return (1-λ)*(L-0.5f);
        }
        float λ = (cdf[ptr]-u)/(cdf[ptr]-cdf[ptr-1]);
        return (1-λ)*(ptr+0.5f) + λ*(ptr-0.5f);
    };

#if defined(USE_CUDA) && defined(__CUDACC__)
    /* Not sure how this should be done by using (thrust::host_vector) ...  */
    thrust::host_vector<float> cdf_h = cdf;
    for(int i=0;i<L;++i){
        float u  = (i+0.5f)*dh;
        float idx= lookup(u);
        icdf[i]  = idx * dh * h_w;
    }
    /* */
    thrust::copy(icdf.begin(), icdf.end(), icdf.begin());
#else
    for(int i=0;i<L;++i){
        float u  = (i+0.5f)*dh;
        float idx= lookup(u);
        icdf[i]  = idx * dh * h_w;
    }
#endif
}
