#include "integral_oneapi.h"
#include <cmath>

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
  const float step = (end - start) / static_cast<float>(count);
  const float area = step * step;
  const float half_step = step * 0.5f;

  float result = 0.0f;
  {
    sycl::buffer<float> result_buf(&result, 1);
    sycl::queue q(device);

    q.submit([&](sycl::handler &cgh) {
       auto reduction = sycl::reduction(result_buf, cgh, sycl::plus<float>());

       cgh.parallel_for(sycl::range<2>(count, count), reduction,
                        [=](sycl::id<2> idx, auto &sum) {
                          float x = start + idx[0] * step + half_step;
                          float y = start + idx[1] * step + half_step;
                          sum += sycl::sin(x) * sycl::cos(y);
                        });
     }).wait();
  }

  return result * area;
}