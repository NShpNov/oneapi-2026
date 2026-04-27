#include "dev_jacobi_oneapi.h"
#include <cmath>

std::vector<float> JacobiDevONEAPI(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy,
        sycl::device device) {

    size_t n = b.size();
    sycl::queue queue(device);

    float* d_a = sycl::malloc_device<float>(a.size(), queue);
    float* d_b = sycl::malloc_device<float>(b.size(), queue);
    float* d_x_old = sycl::malloc_device<float>(n, queue);
    float* d_x_new = sycl::malloc_device<float>(n, queue);
    float* d_error = sycl::malloc_device<float>(1, queue);

    queue.memcpy(d_a, a.data(), a.size() * sizeof(float));
    queue.memcpy(d_b, b.data(), b.size() * sizeof(float));
    queue.memset(d_x_old, 0, n * sizeof(float));
    queue.wait();

    float error = 0.0f;

    for (int iter = 0; iter < ITERATIONS; iter++) {

        queue.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            float sum = 0.0f;

            for (size_t j = 0; j < n; j++) {
                if (j != i) {
                    sum += d_a[i * n + j] * d_x_old[j];
                }
            }

            d_x_new[i] = (d_b[i] - sum) / d_a[i * n + i];
        });

        queue.memset(d_error, 0, sizeof(float));

        queue.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            float diff = sycl::fabs(d_x_new[i] - d_x_old[i]);

            sycl::atomic_ref<float,
                sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space> atom(*d_error);

            float old = atom.load();
            while (old < diff && !atom.compare_exchange_strong(old, diff));
        });

        queue.wait();

        queue.memcpy(&error, d_error, sizeof(float)).wait();

        if (error < accuracy) {
            break;
        }

        queue.memcpy(d_x_old, d_x_new, n * sizeof(float)).wait();
    }
    
    std::vector<float> result(n);
    queue.memcpy(result.data(), d_x_new, n * sizeof(float)).wait();

    sycl::free(d_a, queue);
    sycl::free(d_b, queue);
    sycl::free(d_x_old, queue);
    sycl::free(d_x_new, queue);
    sycl::free(d_error, queue);

    return result;
}