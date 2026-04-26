#include "acc_jacobi_oneapi.h"

#include <cmath>
#include <vector>

std::vector<float> JacobiAccONEAPI(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy,
        sycl::device device) {
    if (accuracy <= 0.0f) {
        accuracy = 1e-6f;
    }

    const std::size_t n = b.size();
    if (n == 0 || a.size() != n * n) {
        return {};
    }

    try {
        sycl::queue q(device);

        std::vector<float> x_current(n, 0.0f);
        std::vector<float> x_next(n, 0.0f);

        sycl::buffer<float, 1> a_buf(a.data(), sycl::range<1>(a.size()));
        sycl::buffer<float, 1> b_buf(b.data(), sycl::range<1>(b.size()));
        sycl::buffer<float, 1> x_current_buf(x_current.data(), sycl::range<1>(n));
        sycl::buffer<float, 1> x_next_buf(x_next.data(), sycl::range<1>(n));

        for (int iter = 0; iter < ITERATIONS; ++iter) {
            q.submit([&](sycl::handler& h) {
                auto a_acc = a_buf.get_access<sycl::access::mode::read>(h);
                auto b_acc = b_buf.get_access<sycl::access::mode::read>(h);
                auto x_current_acc = x_current_buf.get_access<sycl::access::mode::read>(h);
                auto x_next_acc = x_next_buf.get_access<sycl::access::mode::write>(h);

                h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                    const std::size_t i = idx[0];
                    const std::size_t row = i * n;

                    float sum = 0.0f;

                    for (std::size_t j = 0; j < n; ++j) {
                        if (j != i) {
                            sum += a_acc[row + j] * x_current_acc[j];
                        }
                    }

                    x_next_acc[i] = (b_acc[i] - sum) / a_acc[row + i];
                });
            }).wait();

            auto x_current_host = x_current_buf.get_host_access(sycl::read_write);
            auto x_next_host = x_next_buf.get_host_access(sycl::read_only);

            float max_diff = 0.0f;

            for (std::size_t i = 0; i < n; ++i) {
                const float diff = std::fabs(x_next_host[i] - x_current_host[i]);

                if (diff > max_diff) {
                    max_diff = diff;
                }

                x_current_host[i] = x_next_host[i];
            }

            if (max_diff < accuracy) {
                break;
            }
        }

        auto result_host = x_current_buf.get_host_access(sycl::read_only);

        std::vector<float> result(n);
        for (std::size_t i = 0; i < n; ++i) {
            result[i] = result_host[i];
        }

        return result;
    } catch (const sycl::exception&) {
        return {};
    }
}