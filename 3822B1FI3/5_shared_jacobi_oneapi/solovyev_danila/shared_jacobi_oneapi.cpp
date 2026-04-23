#include "shared_jacobi_oneapi.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <sycl/sycl.hpp>

std::vector<float> JacobiSharedONEAPI(const std::vector<float>& a, const std::vector<float>& b, float accuracy, sycl::device device) {
    size_t n = static_cast<size_t>(std::sqrt(a.size()));
    sycl::queue queue(device);

    float* sharedA = sycl::malloc_shared<float>(a.size(), queue);
    float* sharedB = sycl::malloc_shared<float>(n, queue);
    float* sharedXOld = sycl::malloc_shared<float>(n, queue);
    float* sharedXNew = sycl::malloc_shared<float>(n, queue);

    std::copy(a.begin(), a.end(), sharedA);
    std::copy(b.begin(), b.end(), sharedB);
    std::fill(sharedXOld, sharedXOld + n, 0.0f);
    std::fill(sharedXNew, sharedXNew + n, 0.0f);

    for (int k = 0; k < ITERATIONS; ++k) {
        queue.submit([&](sycl::handler& handler) {
            handler.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                int i = idx[0];
                float sum = 0.0f;
                for (int j = 0; j < n; ++j) {
                    if (i != j) {
                        sum += sharedA[i * n + j] * sharedXOld[j];
                    }
                }
                sharedXNew[i] = (sharedB[i] - sum) / sharedA[i * n + i];
                });
            }).wait();

        float maxDiff = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            float diff = std::abs(sharedXNew[i] - sharedXOld[i]);
            if (diff > maxDiff) {
                maxDiff = diff;
            }
        }

        for (size_t i = 0; i < n; ++i) {
            sharedXOld[i] = sharedXNew[i];
        }

        if (maxDiff < accuracy) {
            break;
        }
    }

    std::vector<float> result(sharedXOld, sharedXOld + n);

    sycl::free(sharedA, queue);
    sycl::free(sharedB, queue);
    sycl::free(sharedXOld, queue);
    sycl::free(sharedXNew, queue);

    return result;
}