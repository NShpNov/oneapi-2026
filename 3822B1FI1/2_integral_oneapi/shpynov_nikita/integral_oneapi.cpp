#include "integral_oneapi.h"

#include <cmath>
#include <vector>
#include <algorithm>
#include <cstddef>
#include <stdexcept>
float IntegralONEAPI(float start, float end, int count, sycl::device device) {
    if (count <= 0) return 0.0f;
    if (start > end) std::swap(start, end);

    const size_t n = static_cast<size_t>(count);
    const size_t total = n * n;
    if (total == 0) return 0.0f;

    const float dx = (end - start) / static_cast<float>(count);
    const float area = dx * dx;

    sycl::queue q(device);

    float sum = 0.0f;
    {
        sycl::buffer<float, 1> sum_buf(&sum, sycl::range<1>(1));

        q.submit([&](sycl::handler& h) {
            auto red = sycl::reduction(sum_buf, h, sycl::plus<float>());

            h.parallel_for(sycl::range<1>(total), red, [=](sycl::id<1> idx, auto &acc) {
                const size_t k = idx[0];
                const size_t i = k / n;
                const size_t j = k % n;
                const float x = start + (static_cast<float>(i) + 0.5f) * dx;
                const float y = start + (static_cast<float>(j) + 0.5f) * dx;
                acc += sycl::sin(x) * sycl::cos(y);
            });
        });

        q.wait_and_throw();
    }

    return sum * area;
}