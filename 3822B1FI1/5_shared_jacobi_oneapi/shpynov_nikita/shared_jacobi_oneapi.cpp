#include "shared_jacobi_oneapi.h"

std::vector<float> JacobiSharedONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device) {
    const size_t dim = b.size();
    if (dim == 0) return {};
    if (a.size() != dim * dim) return {};

    sycl::queue q(device);

    float *shared_a = sycl::malloc_shared<float>(dim * dim, q);
    float *shared_b = sycl::malloc_shared<float>(dim, q);
    float *shared_x = sycl::malloc_shared<float>(dim, q);
    float *shared_x_next = sycl::malloc_shared<float>(dim, q);
    if (!shared_a || !shared_b || !shared_x || !shared_x_next) {
        sycl::free(shared_a, q);
        sycl::free(shared_b, q);
        sycl::free(shared_x, q);
        sycl::free(shared_x_next, q);
        throw std::bad_alloc();
    }

    for (size_t i = 0; i < dim * dim; ++i) shared_a[i] = a[i];
    for (size_t i = 0; i < dim; ++i) shared_b[i] = b[i];
    for (size_t i = 0; i < dim; ++i) shared_x[i] = 0.0f;

    std::vector<float> x_host(dim);
    std::vector<float> xnext_host(dim);

    for (int i = 0; i < ITERATIONS; ++i) {
        q.submit([&](sycl::handler &h) {
            h.parallel_for(sycl::range<1>(dim), [=](sycl::id<1> idx) {
                size_t i = idx[0];
                float sum = 0.0f;
                size_t base = i * dim;
                for (size_t j = 0; j < dim; ++j) {
                    if (j == i) continue;
                    sum += shared_a[base + j] * shared_x[j];
                }
                float diag = shared_a[base + i];
                if (diag == 0.0f) {
                    shared_x_next[i] = 0.0f;
                } else {
                    shared_x_next[i] = (shared_b[i] - sum) / diag;
                }
            });
        });

        q.wait();
        float maxdiff = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            x_host[i] = shared_x[i];
            xnext_host[i] = shared_x_next[i];
            float d = std::fabs(xnext_host[i] - x_host[i]);
            if (d > maxdiff) maxdiff = d;
        }

        if (maxdiff < accuracy) {
            std::vector<float> result(dim);
            for (size_t i = 0; i < dim; ++i) 
                result[i] = shared_x_next[i];
            sycl::free(shared_a, q);
            sycl::free(shared_b, q);
            sycl::free(shared_x, q);
            sycl::free(shared_x_next, q);
            return result;
        }
        for (size_t i = 0; i < dim; ++i) shared_x[i] = shared_x_next[i];
    }

    std::vector<float> result(dim);
    for (size_t i = 0; i < dim; ++i)
        result[i] = shared_x_next[i];
    sycl::free(shared_a, q);
    sycl::free(shared_b, q);
    sycl::free(shared_x, q);
    sycl::free(shared_x_next, q);
    return result;
}