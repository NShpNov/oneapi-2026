#include "integral_oneapi.h"

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
    float h = (end - start) / static_cast<float>(count);
    float h2 = h * h;

    sycl::queue q(device);
    sycl::buffer<float> result_buf{0.0f};

    q.submit([&](sycl::handler& h) {
        auto reduction = sycl::reduction(result_buf, h, sycl::plus<float>());
        h.parallel_for(sycl::range<2>{static_cast<std::size_t>(count),
                                      static_cast<std::size_t>(count)},
                       reduction,
                       [=](sycl::id<2> idx, auto& sum) {
                           std::size_t i = idx[0];
                           std::size_t j = idx[1];
                           float x = start + (i + 0.5f) * h;
                           float y = start + (j + 0.5f) * h;
                           float value = sycl::sin(x) * sycl::cos(y);
                           sum += value * h2;
                       });
    });

    auto result_host = result_buf.get_host_access();
    return result_host[0];
}