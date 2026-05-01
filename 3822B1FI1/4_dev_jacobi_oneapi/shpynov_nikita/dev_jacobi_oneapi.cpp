#include "dev_jacobi_oneapi.h"

#include <vector>
#include <cmath>
#include <stdexcept>

#include <memory>

std::vector<float> JacobiDevONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device) {
    const size_t dim = b.size();
    if (dim == 0) return {};
    if (a.size() != dim * dim) return {};
    sycl::queue q(device);

    float *dev_a = sycl::malloc_device<float>(dim * dim, q);
    float *dev_b = sycl::malloc_device<float>(dim, q);
    float *dev_x = sycl::malloc_device<float>(dim, q);
    float *dev_x_next = sycl::malloc_device<float>(dim, q);
    if (!dev_a || !dev_b || !dev_x || !dev_x_next) {
        sycl::free(dev_a, q);
        sycl::free(dev_b, q);
        sycl::free(dev_x, q);
        sycl::free(dev_x_next, q);
        throw std::bad_alloc();
    }
    q.memcpy(dev_a, a.data(), sizeof(float) * dim * dim);
    q.memcpy(dev_b, b.data(), sizeof(float) * dim);

    std::vector<float> zeros(dim, 0.0f);
    q.memcpy(dev_x, zeros.data(), sizeof(float) * dim);

    std::vector<float> x_host(dim);
    std::vector<float> x_next_host(dim);

    for (int i = 0; i < ITERATIONS; ++i) {
        q.submit([&](sycl::handler &h) {
            h.parallel_for(sycl::range<1>(dim), [=](sycl::id<1> idx) {
                size_t i = idx[0];
                float sum = 0.0f;
                size_t base = i * dim;
                for (size_t j = 0; j < dim; ++j) {
                    if (j == i) continue;
                    sum += dev_a[base + j] * dev_x[j];
                }
                float diag = dev_a[base + i];
                if (diag == 0.0f) {
                    dev_x_next[i] = 0.0f;
                } else {
                    dev_x_next[i] = (dev_b[i] - sum) / diag;
                }
            });
        });

        q.wait();
        q.memcpy(x_host.data(), dev_x, sizeof(float) * dim).wait();
        q.memcpy(x_next_host.data(), dev_x_next, sizeof(float) * dim).wait();

        float maxdiff = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            float d = std::fabs(x_next_host[i] - x_host[i]);
            if (d > maxdiff) maxdiff = d;
        }

        if (maxdiff < accuracy) {
            sycl::free(dev_a, q);
            sycl::free(dev_b, q);
            sycl::free(dev_x, q);
            sycl::free(dev_x_next, q);
            return x_next_host;
        }
        q.memcpy(dev_x, dev_x_next, sizeof(float) * dim).wait();
    }
    std::vector<float> result(dim);
    q.memcpy(result.data(), dev_x_next, sizeof(float) * dim).wait();

    sycl::free(dev_a, q);
    sycl::free(dev_b, q);
    sycl::free(dev_x, q);
    sycl::free(dev_x_next, q);

    return result;
}