#include "integral_oneapi.h"
#include <vector>

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
    float delta = (end - start) / static_cast<float>(count);

    float totalSum = 0.0f;

    sycl::queue q(device);
    sycl::buffer<float, 1> sumBuf(&totalSum, sycl::range<1>(1));

    q.submit([&](sycl::handler& h) {
        auto reductionOp = sycl::reduction(sumBuf, h, sycl::plus<>());

        h.parallel_for(sycl::range<2>(count, count), reductionOp,
            [=](sycl::item<2> item, auto& sumAcc) {
                int i = item[0];
                int j = item[1];

                sumAcc += sycl::sin(start + (i + 0.5f) * delta) * sycl::cos(start + (j + 0.5f) * delta);
            });
        }).wait();

    return totalSum * (delta * delta);
}