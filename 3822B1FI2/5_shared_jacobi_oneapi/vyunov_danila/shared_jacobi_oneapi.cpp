#include "shared_jacobi_oneapi.h"

#include <cmath>

std::vector<float> JacobiSharedONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device) {
    const int n = static_cast<int>(b.size());

    sycl::queue q(device);

    float* a_sh    = sycl::malloc_shared<float>(n * n, q);
    float* b_sh    = sycl::malloc_shared<float>(n, q);
    float* x_curr  = sycl::malloc_shared<float>(n, q);
    float* x_next  = sycl::malloc_shared<float>(n, q);

    q.memcpy(a_sh, a.data(), sizeof(float) * n * n).wait();
    q.memcpy(b_sh, b.data(), sizeof(float) * n).wait();

    for (int i = 0; i < n; ++i) {
        x_curr[i] = 0.0f;
    }

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
            int i = static_cast<int>(idx[0]);
            float sum = 0.0f;
            for (int j = 0; j < n; ++j) {
                if (j != i) {
                    sum += a_sh[i * n + j] * x_curr[j];
                }
            }
            x_next[i] = (b_sh[i] - sum) / a_sh[i * n + i];
        }).wait();

        float max_diff = 0.0f;
        for (int i = 0; i < n; ++i) {
            float d = sycl::fabs(x_next[i] - x_curr[i]);
            if (d > max_diff) max_diff = d;
        }

        std::swap(x_curr, x_next);

        if (max_diff < accuracy) break;
    }

    std::vector<float> result(x_curr, x_curr + n);

    sycl::free(a_sh, q);
    sycl::free(b_sh, q);
    sycl::free(x_curr, q);
    sycl::free(x_next, q);

    return result;
}
