#include "dev_jacobi_oneapi.h"

#include <cmath>

std::vector<float> JacobiDevONEAPI(
    const std::vector<float>& a,
    const std::vector<float>& b,
    float accuracy,
    sycl::device device
) {
    const int n = b.size();

    sycl::queue q(device);

    float* a_dev = sycl::malloc_device<float>(n * n, q);
    float* b_dev = sycl::malloc_device<float>(n, q);
    float* x_dev = sycl::malloc_device<float>(n, q);
    float* x_new_dev = sycl::malloc_device<float>(n, q);

    float* diff_arr = sycl::malloc_shared<float>(n, q); 

    q.memcpy(a_dev, a.data(), sizeof(float) * n * n);
    q.memcpy(b_dev, b.data(), sizeof(float) * n);

    q.memset(x_dev, 0, sizeof(float) * n);
    q.memset(x_new_dev, 0, sizeof(float) * n);

    std::vector<float> result(n);

    for (int iter = 0; iter < ITERATIONS; iter++) {

        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            int i = id[0];

            float s = 0.0f;

            for (int j = 0; j < n; j++) {
                if (j != i) {
                    s += a_dev[i * n + j] * x_dev[j];
                }
            }

            float new_val =
                (b_dev[i] - s) / a_dev[i * n + i];

            x_new_dev[i] = new_val;

            float d = new_val - x_dev[i];
            diff_arr[i] = d * d; 
            });

        q.wait();

        float diff = 0.0f;
        for (int i = 0; i < n; i++) {
            diff += diff_arr[i];
        }

        if (std::sqrt(diff) < accuracy) {
            break;
        }

        std::swap(x_dev, x_new_dev);
    }

    q.memcpy(result.data(), x_dev, sizeof(float) * n).wait();

    sycl::free(a_dev, q);
    sycl::free(b_dev, q);
    sycl::free(x_dev, q);
    sycl::free(x_new_dev, q);
    sycl::free(diff_arr, q);

    return result;
}