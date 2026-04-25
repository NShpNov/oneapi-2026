#include "dev_jacobi_oneapi.h"

#include <cmath>
#include <vector>

constexpr size_t BLOCK_DIM = 64;

std::vector<float> JacobiDevONEAPI(
    const std::vector<float>& matrix,
    const std::vector<float>& rhs,
    float tolerance,
    sycl::device dev)
{
    if (tolerance <= 0.0f) {
        tolerance = 1e-6f;
    }

    const size_t matrixSize = rhs.size();
    if (matrixSize == 0 || matrix.size() != matrixSize * matrixSize) {
        return {};
    }

    try {
        sycl::queue execQueue(dev, sycl::property::queue::in_order{});

        float* matDev = sycl::malloc_device<float>(matrix.size(), execQueue);
        float* rhsDev = sycl::malloc_device<float>(matrixSize, execQueue);
        float* xOldDev = sycl::malloc_device<float>(matrixSize, execQueue);
        float* xNewDev = sycl::malloc_device<float>(matrixSize, execQueue);

        if (!matDev || !rhsDev || !xOldDev || !xNewDev) {
            if (matDev) sycl::free(matDev, execQueue);
            if (rhsDev) sycl::free(rhsDev, execQueue);
            if (xOldDev) sycl::free(xOldDev, execQueue);
            if (xNewDev) sycl::free(xNewDev, execQueue);
            return {};
        }

        execQueue.memcpy(matDev, matrix.data(), matrix.size() * sizeof(float));
        execQueue.memcpy(rhsDev, rhs.data(), matrixSize * sizeof(float));
        execQueue.fill(xOldDev, 0.0f, matrixSize);
        execQueue.fill(xNewDev, 0.0f, matrixSize);
        execQueue.wait_and_throw();

        auto roundUp = [](size_t val, size_t align) {
            return (val + align - 1) / align * align;
        };
        size_t globalRange = roundUp(matrixSize, BLOCK_DIM);

        float* xCurrent = xOldDev;
        float* xNext    = xNewDev;

        for (int iter = 0; iter < ITERATIONS; ++iter)
        {
            float maxDiffHost = 0.0f;

            {
                sycl::buffer<float, 1> diffBuffer(&maxDiffHost, sycl::range<1>{1});

                execQueue.submit([&](sycl::handler& cgh)
                {
                    auto maxReduction = sycl::reduction(diffBuffer, cgh, sycl::maximum<float>());

                    cgh.parallel_for(
                        sycl::nd_range<1>(globalRange, BLOCK_DIM),
                        maxReduction,
                        [=](sycl::nd_item<1> item, auto& localMax)
                        {
                            size_t i = item.get_global_id(0);
                            if (i >= matrixSize) return;

                            float rowSum = 0.0f;
                            for (size_t j = 0; j < matrixSize; ++j)
                            {
                                if (j != i)
                                {
                                    rowSum += matDev[i * matrixSize + j] * xCurrent[j];
                                }
                            }

                            float diag = matDev[i * matrixSize + i];
                            float newVal;
                            if (sycl::fabs(diag) < 1e-12f)
                            {
                                newVal = xCurrent[i];
                            }
                            else
                            {
                                newVal = (rhsDev[i] - rowSum) / diag;
                            }

                            xNext[i] = newVal;

                            float diff = sycl::fabs(newVal - xCurrent[i]);
                            localMax.combine(diff);
                        }
                    );
                }).wait();
            }

            std::swap(xCurrent, xNext);

            if (maxDiffHost < tolerance) {
                break;
            }
        }

        std::vector<float> solution(matrixSize);
        execQueue.memcpy(solution.data(), xCurrent, matrixSize * sizeof(float)).wait();

        sycl::free(matDev, execQueue);
        sycl::free(rhsDev, execQueue);
        sycl::free(xOldDev, execQueue);
        sycl::free(xNewDev, execQueue);

        return solution;
    }
    catch (sycl::exception const&) {
        return {};
    }
}
