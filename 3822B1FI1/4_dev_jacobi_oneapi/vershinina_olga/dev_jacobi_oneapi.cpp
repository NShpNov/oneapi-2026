#include "dev_jacobi_oneapi.h"
#include <cmath>
#include <utility>
#include <sycl/sycl.hpp>

std::vector<float> JacobiDevONEAPI(
    const std::vector<float>& a, const std::vector<float>& b,
    float accuracy, sycl::device device) {

    sycl::queue queue(device);
    size_t n = b.size();
    std::vector<float> result(n, 0.0f);

    float* d_a = sycl::malloc_device<float>(n * n, queue);
    float* d_b = sycl::malloc_device<float>(n, queue);
    float* d_x_cur = sycl::malloc_device<float>(n, queue);
    float* d_x_nxt = sycl::malloc_device<float>(n, queue);
    float* d_diff = sycl::malloc_shared<float>(1, queue);

    queue.fill(d_x_cur, 0.0f, n);
    queue.memcpy(d_a, a.data(), sizeof(float) * n * n);
    queue.memcpy(d_b, b.data(), sizeof(float) * n);
    queue.wait();

    float* cur_ptr = d_x_cur;
    float* nxt_ptr = d_x_nxt;

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        *d_diff = 0.0f;

        queue.submit([&](sycl::handler& cgh) {
            auto red = sycl::reduction(d_diff, sycl::maximum<float>());

            cgh.parallel_for(sycl::range<1>(n), red, [=](sycl::id<1> idx, auto& max_val) {
                size_t i = idx[0];
                float sum = 0.0f;
                for (size_t j = 0; j < n; ++j) {
                    if (j != i) {
                        sum += d_a[i * n + j] * cur_ptr[j];
                    }
                }
                nxt_ptr[i] = (d_b[i] - sum) / d_a[i * n + i];
                max_val.combine(sycl::fabs(nxt_ptr[i] - cur_ptr[i]));
                });
            }).wait();

            float current_diff = *d_diff;
            if (current_diff < accuracy) {
                std::swap(cur_ptr, nxt_ptr);
                break;
            }
            std::swap(cur_ptr, nxt_ptr);
    }

    queue.memcpy(result.data(), cur_ptr, sizeof(float) * n).wait();

    sycl::free(d_a, queue);
    sycl::free(d_b, queue);
    sycl::free(d_x_cur, queue);
    sycl::free(d_x_nxt, queue);
    sycl::free(d_diff, queue);

    return result;
}