#include "dev_jacobi_oneapi.h"
#include <algorithm>
#include <cmath>

std::vector<float> JacobiDevONEAPI(const std::vector<float> a,
                                   const std::vector<float> b, float accuracy,
                                   sycl::device device) {

  const int n = static_cast<int>(b.size());

  sycl::queue queue(device, sycl::property::queue::in_order{});

  std::vector<float> inv_diag(n);
  for (int i = 0; i < n; ++i) {
    inv_diag[i] = 1.0f / a[i * n + i];
  }

  float *d_a = sycl::malloc_device<float>(n * n, queue);
  float *d_b = sycl::malloc_device<float>(n, queue);
  float *d_inv_d = sycl::malloc_device<float>(n, queue);
  float *d_x = sycl::malloc_device<float>(n, queue);
  float *d_x_new = sycl::malloc_device<float>(n, queue);
  float *d_max_diff = sycl::malloc_shared<float>(1, queue);

  queue.memcpy(d_a, a.data(), sizeof(float) * n * n);
  queue.memcpy(d_b, b.data(), sizeof(float) * n);
  queue.memcpy(d_inv_d, inv_diag.data(), sizeof(float) * n);
  queue.fill(d_x, 0.0f, n);

  const size_t wg_size = 128;
  const size_t global_size = ((n + wg_size - 1) / wg_size) * wg_size;

  for (int iter = 0; iter < ITERATIONS; ++iter) {

    queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(global_size, wg_size),
                       [=](sycl::nd_item<1> item) {
                         size_t i = item.get_global_id(0);
                         if (i >= static_cast<size_t>(n))
                           return;

                         const size_t row = i * n;
                         float sum = 0.0f;
                         for (int j = 0; j < n; ++j) {
                           if (j != static_cast<int>(i)) {
                             sum += d_a[row + j] * d_x[j];
                           }
                         }
                         d_x_new[i] = d_inv_d[i] * (d_b[i] - sum);
                       });
    });

    d_max_diff[0] = 0.0f;

    queue
        .submit([&](sycl::handler &cgh) {
          auto reduction = sycl::reduction(d_max_diff, sycl::maximum<float>());
          cgh.parallel_for(sycl::nd_range<1>(global_size, wg_size), reduction,
                           [=](sycl::nd_item<1> item, auto &reducer) {
                             size_t i = item.get_global_id(0);
                             if (i >= static_cast<size_t>(n))
                               return;
                             float diff = std::fabs(d_x_new[i] - d_x[i]);
                             reducer.combine(diff);
                           });
        })
        .wait();

    if (d_max_diff[0] < accuracy) {
      std::swap(d_x, d_x_new);
      break;
    }

    std::swap(d_x, d_x_new);
  }

  std::vector<float> result(n);
  queue.memcpy(result.data(), d_x, sizeof(float) * n).wait();

  sycl::free(d_a, queue);
  sycl::free(d_b, queue);
  sycl::free(d_inv_d, queue);
  sycl::free(d_x, queue);
  sycl::free(d_x_new, queue);
  sycl::free(d_max_diff, queue);

  return result;
}