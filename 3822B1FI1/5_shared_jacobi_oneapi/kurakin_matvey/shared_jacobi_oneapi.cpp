#include "shared_jacobi_oneapi.h"
#include <cmath>

std::vector<float> JacobiSharedONEAPI(const std::vector<float> &a,
                                      const std::vector<float> &b,
                                      float accuracy, sycl::device device) {

  const size_t n = b.size();

  sycl::queue queue(device, sycl::property::queue::in_order{});

  float *mat_a = sycl::malloc_shared<float>(a.size(), queue);
  float *mat_b = sycl::malloc_shared<float>(b.size(), queue);
  float *x = sycl::malloc_shared<float>(n, queue);
  float *x_new = sycl::malloc_shared<float>(n, queue);
  float *max_diff = sycl::malloc_shared<float>(1, queue);

  queue.memcpy(mat_a, a.data(), a.size() * sizeof(float));
  queue.memcpy(mat_b, b.data(), b.size() * sizeof(float));
  queue.fill(x, 0.0f, n);
  queue.wait();

  const size_t wg_size = 64;
  const size_t global_size = ((n + wg_size - 1) / wg_size) * wg_size;

  for (int iter = 0; iter < ITERATIONS; ++iter) {

    *max_diff = 0.0f;

    auto reduction = sycl::reduction(max_diff, sycl::maximum<float>());

    queue.parallel_for(sycl::nd_range<1>(global_size, wg_size), reduction,
                       [=](sycl::nd_item<1> item, auto &reducer) {
                         size_t i = item.get_global_id(0);
                         if (i >= n)
                           return;

                         const size_t row = i * n;
                         float sum = 0.0f;
                         for (size_t j = 0; j < n; ++j) {
                           if (j != i) {
                             sum += mat_a[row + j] * x[j];
                           }
                         }
                         float value = (mat_b[i] - sum) / mat_a[row + i];
                         x_new[i] = value;

                         float diff = std::fabs(value - x[i]);
                         reducer.combine(diff);
                       });

    queue.wait();

    if (*max_diff < accuracy) {
      std::swap(x, x_new);
      break;
    }

    std::swap(x, x_new);
  }

  std::vector<float> result(n);
  queue.memcpy(result.data(), x, n * sizeof(float)).wait();

  sycl::free(mat_a, queue);
  sycl::free(mat_b, queue);
  sycl::free(x, queue);
  sycl::free(x_new, queue);
  sycl::free(max_diff, queue);

  return result;
}