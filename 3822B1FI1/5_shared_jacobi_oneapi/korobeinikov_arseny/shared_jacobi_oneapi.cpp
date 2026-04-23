#include "shared_jacobi_oneapi.h"

#include <vector>

std::vector<float> JacobiSharedONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device) {
    const std::size_t n = b.size();

    sycl::queue queue(device);

    float* a_shared = sycl::malloc_shared<float>(a.size(), queue);
    float* b_shared = sycl::malloc_shared<float>(b.size(), queue);
    float* current = sycl::malloc_shared<float>(n, queue);
    float* next = sycl::malloc_shared<float>(n, queue);
    float* max_diff = sycl::malloc_shared<float>(1, queue);

    queue.memcpy(a_shared, a.data(), sizeof(float) * a.size());
    queue.memcpy(b_shared, b.data(), sizeof(float) * b.size());
    queue.fill(current, 0.0f, n).wait();

    const std::size_t local_size = 128;
    const std::size_t global_size =
        ((n + local_size - 1) / local_size) * local_size;

    for (int iteration = 0; iteration < ITERATIONS; ++iteration) {
        max_diff[0] = 0.0f;

        queue.submit([&](sycl::handler& handler) {
            auto reduction = sycl::reduction(max_diff, sycl::maximum<float>());

            handler.parallel_for(sycl::nd_range<1>(global_size, local_size), reduction,
                                 [=](sycl::nd_item<1> item, auto& diff_reducer) {
                                     std::size_t i = item.get_global_id(0);
                                     if (i >= n) {
                                         return;
                                     }

                                     const std::size_t row = i * n;
                                     float sum = 0.0f;

                                     for (std::size_t j = 0; j < n; ++j) {
                                         if (j != i) {
                                             sum += a_shared[row + j] * current[j];
                                         }
                                     }

                                     float value = (b_shared[i] - sum) / a_shared[row + i];
                                     next[i] = value;

                                     float diff = sycl::fabs(value - current[i]);
                                     diff_reducer.combine(diff);
                                 });
        }).wait();

        std::swap(current, next);

        if (max_diff[0] < accuracy) {
            break;
        }
    }

    std::vector<float> result(n);
    for (std::size_t i = 0; i < n; ++i) {
        result[i] = current[i];
    }

    sycl::free(a_shared, queue);
    sycl::free(b_shared, queue);
    sycl::free(current, queue);
    sycl::free(next, queue);
    sycl::free(max_diff, queue);

    return result;
}
