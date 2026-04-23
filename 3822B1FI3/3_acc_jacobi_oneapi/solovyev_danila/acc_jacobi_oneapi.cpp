#include "acc_jacobi_oneapi.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

std::vector<float> JacobiAccONEAPI(const std::vector<float>& a, const std::vector<float>& b, float accuracy, sycl::device device) {
    size_t n = static_cast<size_t>(std::sqrt(a.size()));

    std::vector<float> xOld(n, 0.0f);
    std::vector<float> xNew(n, 0.0f);

    sycl::queue q(device);

    sycl::buffer<float, 1> bufA(a.data(), sycl::range<1>(a.size()));
    sycl::buffer<float, 1> bufB(b.data(), sycl::range<1>(n));

    sycl::buffer<float, 1> bufXCurr(xOld.data(), sycl::range<1>(n));
    sycl::buffer<float, 1> bufXNext(xNew.data(), sycl::range<1>(n));

    for (int k = 0; k < ITERATIONS; ++k) {
        q.submit([&](sycl::handler& h) {
            auto accA = bufA.get_access<sycl::access::mode::read>(h);
            auto accB = bufB.get_access<sycl::access::mode::read>(h);

            auto accCurr = bufXCurr.get_access<sycl::access::mode::read>(h);
            auto accNext = bufXNext.get_access<sycl::access::mode::write>(h);

            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                int i = idx[0];
                float sum = 0.0f;
                for (int j = 0; j < n; ++j) {
                    if (i != j) {
                        sum += accA[i * n + j] * accCurr[j];
                    }
                }
                accNext[i] = (accB[i] - sum) / accA[i * n + i];
                });
            });

        float maxDiff = 0.0f;
        {
            auto hostNew = bufXNext.get_access<sycl::access::mode::read>();
            auto hostOld = bufXCurr.get_access<sycl::access::mode::read>();

            for (size_t i = 0; i < n; ++i) {
                float diff = std::abs(hostNew[i] - hostOld[i]);
                if (diff > maxDiff) {
                    maxDiff = diff;
                }
            }
        }

        {
            auto hostNew = bufXNext.get_access<sycl::access::mode::read>();
            auto hostOld = bufXCurr.get_access<sycl::access::mode::write>();
            for (size_t i = 0; i < n; ++i) {
                hostOld[i] = hostNew[i];
            }
        }

        if (maxDiff < accuracy) {
            break;
        }
    }

    return xOld;
}