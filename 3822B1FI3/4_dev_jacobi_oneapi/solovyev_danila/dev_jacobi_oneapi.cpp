#include "dev_jacobi_oneapi.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <sycl/sycl.hpp>

std::vector<float> JacobiDevONEAPI(const std::vector<float>& a, const std::vector<float>& b, float accuracy, sycl::device device) {
    size_t n = static_cast<size_t>(std::sqrt(a.size()));
    sycl::queue queue(device);

    float* devA = sycl::malloc_device<float>(a.size(), queue);
    float* devB = sycl::malloc_device<float>(n, queue);

    float* devXOld = sycl::malloc_device<float>(n, queue);
    float* devXNew = sycl::malloc_device<float>(n, queue);

    std::vector<float> xOldHost(n, 0.0f);
    std::vector<float> xNewHost(n, 0.0f);

    queue.memcpy(devA, a.data(), a.size() * sizeof(float)).wait();
    queue.memcpy(devB, b.data(), n * sizeof(float)).wait();
    queue.memcpy(devXOld, xOldHost.data(), n * sizeof(float)).wait();

    for (int k = 0; k < ITERATIONS; ++k) {
        queue.submit([&](sycl::handler& handler) {
            handler.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                int i = idx[0];
                float sum = 0.0f;
                for (int j = 0; j < n; ++j) {
                    if (i != j) {
                        sum += devA[i * n + j] * devXOld[j];
                    }
                }
                devXNew[i] = (devB[i] - sum) / devA[i * n + i];
                });
            }).wait();

        queue.memcpy(xNewHost.data(), devXNew, n * sizeof(float)).wait();

        float maxDiff = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            float diff = std::abs(xNewHost[i] - xOldHost[i]);
            if (diff > maxDiff) {
                maxDiff = diff;
            }
        }

        xOldHost = xNewHost;
        queue.memcpy(devXOld, xOldHost.data(), n * sizeof(float)).wait();

        if (maxDiff < accuracy) {
            break;
        }
    }

    sycl::free(devA, queue);
    sycl::free(devB, queue);
    sycl::free(devXOld, queue);
    sycl::free(devXNew, queue);

    return xOldHost;
}