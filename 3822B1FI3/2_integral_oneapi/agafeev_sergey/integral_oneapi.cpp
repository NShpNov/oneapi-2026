#include "integral_oneapi.h"
#include <cmath>

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
    sycl::queue queue(device);

    float result = 0.0f;

    float step = (end - start) / count;

    sycl::buffer<float> result_buf(&result, 1);

    queue.submit([&](sycl::handler& h) {
        auto reduction = sycl::reduction(result_buf, h, std::plus<float>());

        h.parallel_for(
            sycl::range<2>(count, count),
            reduction,
            [=](sycl::id<2> idx, auto& sum) {
                int i = idx[0];
                int j = idx[1];

                float x = start + (i + 0.5f) * step;
                float y = start + (j + 0.5f) * step;

                float value = sycl::sin(x) * sycl::cos(y);

                sum += value * step * step;
            }
        );
    });

    queue.wait();

    return result;
}