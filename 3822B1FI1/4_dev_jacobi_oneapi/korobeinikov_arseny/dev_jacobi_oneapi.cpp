#include "dev_jacobi_oneapi.h"

#include <cmath>
#include <vector>

std::vector<float> JacobiDevONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device) {
    const std::size_t n = b.size();

    sycl::queue queue(device);

    float* a_dev = sycl::malloc_device<float>(a.size(), queue);
    float* b_dev = sycl::malloc_device<float>(b.size(), queue);
    float* x_dev = sycl::malloc_device<float>(n, queue);
    float* next_dev = sycl::malloc_device<float>(n, queue);
    float* diff_dev = sycl::malloc_shared<float>(1, queue);

    queue.memcpy(a_dev, a.data(), sizeof(float) * a.size());
    queue.memcpy(b_dev, b.data(), sizeof(float) * b.size());
    queue.fill(x_dev, 0.0f, n).wait();

    const std::size_t local_size = 128;
    const std::size_t global_size =
        ((n + local_size - 1) / local_size) * local_size;

    bool converged = false;

    for (int iteration = 0; iteration < ITERATIONS; ++iteration) {
        queue.parallel_for(sycl::nd_range<1>(global_size, local_size),
                           [=](sycl::nd_item<1> item) {
                               std::size_t i = item.get_global_id(0);
                               if (i >= n) {
                                   return;
                               }

                               const std::size_t row = i * n;
                               float sum = 0.0f;

                               for (std::size_t j = 0; j < n; ++j) {
                                   if (j != i) {
                                       sum += a_dev[row + j] * x_dev[j];
                                   }
                               }

                               next_dev[i] = (b_dev[i] - sum) / a_dev[row + i];
                           });

        diff_dev[0] = 0.0f;

        queue.submit([&](sycl::handler& handler) {
            auto reduction = sycl::reduction(diff_dev, sycl::maximum<float>());

            handler.parallel_for(sycl::nd_range<1>(global_size, local_size), reduction,
                                 [=](sycl::nd_item<1> item, auto& max_diff) {
                                     std::size_t i = item.get_global_id(0);
                                     if (i >= n) {
                                         return;
                                     }

                                     float diff = sycl::fabs(next_dev[i] - x_dev[i]);
                                     max_diff.combine(diff);
                                 });
        }).wait();

        std::swap(x_dev, next_dev);

        if (diff_dev[0] < accuracy) {
            converged = true;
            break;
        }
    }

    std::vector<float> result(n);
    queue.memcpy(result.data(), x_dev, sizeof(float) * n).wait();

    sycl::free(a_dev, queue);
    sycl::free(b_dev, queue);
    sycl::free(x_dev, queue);
    sycl::free(next_dev, queue);
    sycl::free(diff_dev, queue);

    return result;
}
