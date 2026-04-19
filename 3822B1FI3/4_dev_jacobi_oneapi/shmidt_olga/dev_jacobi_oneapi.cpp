#include "dev_jacobi_oneapi.h"
#include <cmath>

std::vector<float> JacobiDevONEAPI(
    const std::vector<float>& a, const std::vector<float>& b,
    float accuracy, sycl::device device)
{
    size_t n = b.size();
    std::vector<float> x(n, 0.0f);
    std::vector<float> x_new(n, 0.0f);

    sycl::queue queue(device);

    float* d_a = sycl::malloc_device<float>(n * n, queue);
    float* d_b = sycl::malloc_device<float>(n, queue);
    float* d_x = sycl::malloc_device<float>(n, queue);
    float* d_x_new = sycl::malloc_device<float>(n, queue);

    queue.memcpy(d_a, a.data(), n * n * sizeof(float));
    queue.memcpy(d_b, b.data(), n * sizeof(float));
    queue.memcpy(d_x, x.data(), n * sizeof(float));
    queue.wait();

    for (int iter = 0; iter < ITERATIONS; ++iter)
    {
        queue.submit([&](sycl::handler& cgh)
            {
                cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx)
                    {
                        size_t i = idx[0];
                        float sum = d_b[i];

                        for (size_t j = 0; j < n; ++j)
                        {
                            if (i != j)
                            {
                                sum -= d_a[i * n + j] * d_x[j];
                            }
                        }

                        d_x_new[i] = sum / d_a[i * n + i];
                    });
            });
        queue.wait();

        queue.memcpy(x_new.data(), d_x_new, n * sizeof(float));
        queue.wait();

        float max_diff = 0.0f;
        for (size_t i = 0; i < n; ++i)
        {
            float diff = std::fabs(x_new[i] - x[i]);
            if (diff > max_diff)
            {
                max_diff = diff;
            }
            x[i] = x_new[i];
        }

        queue.memcpy(d_x, x.data(), n * sizeof(float));
        queue.wait();

        if (max_diff < accuracy)
        {
            break;
        }
    }

    sycl::free(d_a, queue);
    sycl::free(d_b, queue);
    sycl::free(d_x, queue);
    sycl::free(d_x_new, queue);

    return x;
}