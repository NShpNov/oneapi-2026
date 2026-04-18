#include "integral_oneapi.h"
#include <cmath>

float IntegralONEAPI(float start, float end, int count, sycl::device device)
{
    float step = (end - start) / count;
    float total_sum = 0.0f;

    sycl::queue queue(device);

    {
        sycl::buffer<float> sum_buf(&total_sum, 1);

        queue.submit([&](sycl::handler& cgh)
            {
                auto sum_reduction = sycl::reduction(sum_buf, cgh, sycl::plus<>());

                cgh.parallel_for(sycl::range<2>(count, count), sum_reduction,
                    [=](sycl::item<2> item, auto& sum)
                    {
                        int i = item.get_id(0);
                        int j = item.get_id(1);

                        float x = start + (i + 0.5f) * step;
                        float y = start + (j + 0.5f) * step;

                        sum += sycl::sin(x) * sycl::cos(y) * step * step;
                    });
            });
    }

    return total_sum;
}