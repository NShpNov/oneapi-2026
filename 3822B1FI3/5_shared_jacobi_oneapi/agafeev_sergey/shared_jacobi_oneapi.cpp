#include "shared_jacobi_oneapi.h"
#include <cmath>

std::vector<float> JacobiSharedONEAPI(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy,
        sycl::device device) {

    size_t n = b.size();
    sycl::queue queue(device);

    float* s_a = sycl::malloc_shared<float>(a.size(), queue);
    float* s_b = sycl::malloc_shared<float>(b.size(), queue);
    float* x_old = sycl::malloc_shared<float>(n, queue);
    float* x_new = sycl::malloc_shared<float>(n, queue);
    float* error = sycl::malloc_shared<float>(1, queue);

    for (size_t i = 0; i < a.size(); i++) s_a[i] = a[i];
    for (size_t i = 0; i < b.size(); i++) s_b[i] = b[i];

    for (size_t i = 0; i < n; i++) x_old[i] = 0.0f;

    for (int iter = 0; iter < ITERATIONS; iter++) {

        queue.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            float sum = 0.0f;

            for (size_t j = 0; j < n; j++) {
                if (j != i) {
                    sum += s_a[i * n + j] * x_old[j];
                }
            }

            x_new[i] = (s_b[i] - sum) / s_a[i * n + i];
        });

        *error = 0.0f;

        queue.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            float diff = sycl::fabs(x_new[i] - x_old[i]);

            sycl::atomic_ref<float,
                sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space> atom(*error);

            float old = atom.load();
            while (old < diff && !atom.compare_exchange_strong(old, diff));
        });

        queue.wait();

        if (*error < accuracy) {
            break;
        }

        for (size_t i = 0; i < n; i++) {
            x_old[i] = x_new[i];
        }
    }

    std::vector<float> result(n);
    for (size_t i = 0; i < n; i++) {
        result[i] = x_new[i];
    }

    sycl::free(s_a, queue);
    sycl::free(s_b, queue);
    sycl::free(x_old, queue);
    sycl::free(x_new, queue);
    sycl::free(error, queue);

    return result;
}