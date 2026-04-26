#include "shared_jacobi_oneapi.h"
#include <cmath>

std::vector<float> JacobiSharedONEAPI(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy,
        sycl::device device) {

    const size_t dim = b.size();

    sycl::queue computeQueue(device, sycl::property::queue::in_order{});

    std::vector<float> result(dim, 0.0f);

    float* devMatrixA = sycl::malloc_shared<float>(a.size(), computeQueue);
    float* devRhs     = sycl::malloc_shared<float>(b.size(), computeQueue);
    float* devXcurr   = sycl::malloc_shared<float>(dim, computeQueue);
    float* devXnext   = sycl::malloc_shared<float>(dim, computeQueue);
    float* devMaxDiff = sycl::malloc_shared<float>(1, computeQueue);

    computeQueue.memcpy(devMatrixA, a.data(), a.size() * sizeof(float));
    computeQueue.memcpy(devRhs,     b.data(), b.size() * sizeof(float));
    computeQueue.memset(devXcurr,   0, sizeof(float) * dim);
    computeQueue.memset(devXnext,   0, sizeof(float) * dim);
    computeQueue.wait();

    for (int iteration = 0; iteration < ITERATIONS; ++iteration) {

        *devMaxDiff = 0.0f;

        auto maxReducer = sycl::reduction(
            devMaxDiff,
            sycl::maximum<float>());

        computeQueue.parallel_for(
            sycl::range<1>(dim),
            maxReducer,
            [=](sycl::id<1> idx, auto& diff) {

                size_t i = idx[0];
                float rowSum = devRhs[i];
                const size_t rowOffset = i * dim;

                #pragma unroll 4
                for (size_t j = 0; j < dim; ++j) {
                    if (j != i) {
                        rowSum -= devMatrixA[rowOffset + j] * devXcurr[j];
                    }
                }

                float newX = rowSum / devMatrixA[rowOffset + i];
                devXnext[i] = newX;

                float delta = sycl::fabs(newX - devXcurr[i]);
                diff.combine(delta);
            });

        computeQueue.wait();

        if (*devMaxDiff < accuracy) {
            std::swap(devXcurr, devXnext);
            break;
        }

        std::swap(devXcurr, devXnext);
    }

    computeQueue.memcpy(result.data(), devXcurr, dim * sizeof(float)).wait();

    sycl::free(devMatrixA, computeQueue);
    sycl::free(devRhs,     computeQueue);
    sycl::free(devXcurr,   computeQueue);
    sycl::free(devXnext,   computeQueue);
    sycl::free(devMaxDiff, computeQueue);

    return result;
}