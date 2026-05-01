#include "acc_jacobi_oneapi.h"

#include <vector>
#include <cmath>
#include <stdexcept>

std::vector<float> JacobiAccONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device) {
    const size_t dim = b.size();
    if (dim == 0) return {};
    if (a.size() != dim * dim) return {};
    sycl::queue q(device);
    sycl::buffer<float, 1> buf_a(a.data(), sycl::range<1>(dim * dim));
    sycl::buffer<float, 1> buf_b(b.data(), sycl::range<1>(dim));
    sycl::buffer<float, 1> buf_x{sycl::range<1>(dim)};
    sycl::buffer<float, 1> buf_x_next{sycl::range<1>(dim)};
    {
        auto acc_x = buf_x.get_access<sycl::access::mode::write>();
        for (size_t i = 0; i < dim; ++i)
            acc_x[i] = 0.0f;
    }

    std::vector<float> x_host(dim);
    std::vector<float> x_next_host(dim);

    for (int i = 0; i < ITERATIONS; ++i) {
        q.submit([&](sycl::handler &h) {
            auto acc_a = buf_a.get_access<sycl::access::mode::read>(h);
            auto acc_b = buf_b.get_access<sycl::access::mode::read>(h);
            auto acc_x = buf_x.get_access<sycl::access::mode::read>(h);
            auto acc_xn = buf_x_next.get_access<sycl::access::mode::write>(h);

            h.parallel_for(sycl::range<1>(dim), [=](sycl::id<1> idx) {
                const size_t i = idx[0];
                float sum = 0.0f;
                const size_t base = i * dim;
                for (size_t j = 0; j < dim; ++j) {
                    if (j == i) continue;
                    sum += acc_a[base + j] * acc_x[j];
                }
                float diag = acc_a[base + i];
                if (diag == 0.0f) {
                    acc_xn[i] = 0.0f;
                } else {
                    acc_xn[i] = (acc_b[i] - sum) / diag;
                }
            });
        });

        q.wait();
        {
            auto acc_x_h = buf_x.get_access<sycl::access::mode::read>();
            auto acc_xn_h = buf_x_next.get_access<sycl::access::mode::read>();
            float maxdiff = 0.0f;
            for (size_t i = 0; i < dim; ++i) {
                x_host[i] = acc_x_h[i];
                x_next_host[i] = acc_xn_h[i];
                float d = std::fabs(x_next_host[i] - x_host[i]);
                if (d > maxdiff) maxdiff = d;
            }

            if (maxdiff < accuracy) {
                return x_next_host;
            }
        }

        {
            auto acc_x_w = buf_x.get_access<sycl::access::mode::write>();
            for (size_t i = 0; i < dim; ++i) acc_x_w[i] = x_next_host[i];
        }
    }
    {
        std::vector<float> result(dim);
        auto acc_xn_h = buf_x_next.get_access<sycl::access::mode::read>();
        for (size_t i = 0; i < dim; ++i) result[i] = acc_xn_h[i];
        return result;
    }
}