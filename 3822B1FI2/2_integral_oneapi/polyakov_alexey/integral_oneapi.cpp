#include "integral_oneapi.h"
#include <cmath>

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
  float step = (end - start) / count;
  float result = 0.0f;
  
  sycl::queue queue(device);
  
  {
    sycl::buffer<float> result_buf(&result, 1);
    
    queue.submit([&](sycl::handler &handler) {
      auto reduction = sycl::reduction(result_buf, handler, sycl::plus<>());
      
      handler.parallel_for(
        sycl::range<2>(count, count),
        reduction,
        [=](sycl::id<2> idx, auto &partial_sum) {
          int i = idx[0];
          int j = idx[1];
          
          float x = start + (i + 0.5f) * step;
          float y = start + (j + 0.5f) * step;
          
          float function_value = std::sin(x) * std::cos(y);
          float cell_area = step * step;
          
          partial_sum += function_value * cell_area;
        }
      );
    }).wait();
  }
  
  return result;
}