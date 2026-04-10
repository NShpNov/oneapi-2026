#include "acc_jacobi_oneapi.h"
#include <algorithm>
#include <cmath>

std::vector<float> JacobiAccONEAPI(const std::vector<float> a,
                                   const std::vector<float> b, float accuracy,
                                   sycl::device device) {

  const size_t n = static_cast<size_t>(std::sqrt(a.size()));

  std::vector<float> inv_diag(n);
  for (size_t i = 0; i < n; ++i) {
    inv_diag[i] = 1.0f / a[i * n + i];
  }

  sycl::queue q(device, sycl::property::queue::in_order{});

  sycl::buffer<float, 1> a_buf(a.data(), sycl::range<1>(a.size()));
  sycl::buffer<float, 1> b_buf(b.data(), sycl::range<1>(b.size()));
  sycl::buffer<float, 1> inv_diag_buf(inv_diag.data(), sycl::range<1>(n));

  sycl::buffer<float, 1> x_curr_buf{n};
  sycl::buffer<float, 1> x_next_buf{n};
  sycl::buffer<float, 1> max_diff_buf{1};

  q.submit([&](sycl::handler &cgh) {
    auto x_acc = x_curr_buf.get_access<sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for(sycl::range<1>(n),
                     [=](sycl::id<1> idx) { x_acc[idx] = 0.0f; });
  });

  for (int iter = 0; iter < ITERATIONS; ++iter) {
    q.submit([&](sycl::handler &cgh) {
      auto a_acc = a_buf.get_access<sycl::access::mode::read>(cgh);
      auto b_acc = b_buf.get_access<sycl::access::mode::read>(cgh);
      auto inv_d_acc = inv_diag_buf.get_access<sycl::access::mode::read>(cgh);
      auto x_curr_acc = x_curr_buf.get_access<sycl::access::mode::read>(cgh);
      auto x_next_acc = x_next_buf.get_access<sycl::access::mode::write>(cgh);

      cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
        const size_t i = idx[0];
        const size_t row_off = i * n;
        float sum = 0.0f;
        for (size_t j = 0; j < n; ++j) {
          if (j != i) {
            sum += a_acc[row_off + j] * x_curr_acc[j];
          }
        }
        x_next_acc[i] = inv_d_acc[i] * (b_acc[i] - sum);
      });
    });

    q.submit([&](sycl::handler &cgh) {
      auto max_acc =
          max_diff_buf.get_access<sycl::access::mode::discard_write>(cgh);
      cgh.single_task([=]() { max_acc[0] = 0.0f; });
    });

    q.submit([&](sycl::handler &cgh) {
       auto x_curr_acc = x_curr_buf.get_access<sycl::access::mode::read>(cgh);
       auto x_next_acc = x_next_buf.get_access<sycl::access::mode::read>(cgh);
       auto reduction =
           sycl::reduction(max_diff_buf, cgh, sycl::maximum<float>());
       cgh.parallel_for(sycl::range<1>(n), reduction,
                        [=](sycl::id<1> idx, auto &reducer) {
                          const size_t i = idx[0];
                          float diff = std::fabs(x_next_acc[i] - x_curr_acc[i]);
                          reducer.combine(diff);
                        });
     }).wait();

    if (max_diff_buf.get_host_access()[0] < accuracy) {
      break;
    }

    q.submit([&](sycl::handler &cgh) {
      auto src = x_next_buf.get_access<sycl::access::mode::read>(cgh);
      auto dst = x_curr_buf.get_access<sycl::access::mode::write>(cgh);
      cgh.copy(src, dst);
    });
  }

  std::vector<float> result(n);
  q.submit([&](sycl::handler &cgh) {
     auto x_acc = x_curr_buf.get_access<sycl::access::mode::read>(cgh);
     cgh.copy(x_acc, result.data());
   }).wait();

  return result;
}