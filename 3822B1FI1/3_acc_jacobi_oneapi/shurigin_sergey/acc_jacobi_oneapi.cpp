#include "acc_jacobi_oneapi.h"
#include <cmath>

std::vector<float> JacobiAccONEAPI(
        const std::vector<float>& a, const std::vector<float>& b, 
        float accuracy, sycl::device device) {
    
    const size_t n = b.size();
    const float accuracy_sq = accuracy * accuracy;


    std::vector<float> inv_diag(n);
    for (size_t i = 0; i < n; i++) {
        inv_diag[i] = 1.0f / a[i * n + i];
    }

    sycl::queue q(device, {sycl::property::queue::in_order{}});

    sycl::buffer<float, 1> a_buf(a.data(), sycl::range<1>(a.size()));
    sycl::buffer<float, 1> b_buf(b.data(), sycl::range<1>(b.size()));
    sycl::buffer<float, 1> inv_diag_buf(inv_diag.data(), sycl::range<1>(n));
    sycl::buffer<float, 1> x_curr_buf(sycl::range<1>(n));
    sycl::buffer<float, 1> x_next_buf(sycl::range<1>(n));
    sycl::buffer<float, 1> norm_buf(sycl::range<1>(1));

    q.submit([&](sycl::handler& cgh) {
        auto x_acc = x_curr_buf.get_access<sycl::access::mode::write>(cgh);
        cgh.fill(x_acc, 0.0f);
    });

    for (int iter = 0; iter < ITERATIONS; iter++) {
        q.submit([&](sycl::handler& cgh) {
            auto a_acc = a_buf.get_access<sycl::access::mode::read>(cgh);
            auto b_acc = b_buf.get_access<sycl::access::mode::read>(cgh);
            auto inv_diag_acc = inv_diag_buf.get_access<sycl::access::mode::read>(cgh);
            auto x_curr_acc = x_curr_buf.get_access<sycl::access::mode::read>(cgh);
            auto x_next_acc = x_next_buf.get_access<sycl::access::mode::write>(cgh);

            cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                size_t i = idx[0];
                float sum = 0.0f;
                for (size_t j = 0; j < n; ++j) {
                    if (j != i) sum += a_acc[i * n + j] * x_curr_acc[j];
                }
                x_next_acc[i] = inv_diag_acc[i] * (b_acc[i] - sum);
            });
        });

        q.submit([&](sycl::handler& cgh) {
            auto x_curr_acc = x_curr_buf.get_access<sycl::access::mode::read>(cgh);
            auto x_next_acc = x_next_buf.get_access<sycl::access::mode::read>(cgh);
            auto reduction = sycl::reduction(norm_buf, cgh, sycl::plus<float>());

            cgh.parallel_for(sycl::range<1>(n), reduction,
                [=](sycl::id<1> idx, auto& norm_red) {
                    size_t i = idx[0];
                    float diff = x_next_acc[i] - x_curr_acc[i];
                    norm_red += diff * diff; 
                });
        });

        float current_norm_sq;
        q.submit([&](sycl::handler& cgh) {
            auto n_acc = norm_buf.get_access<sycl::access::mode::read>(cgh);
            cgh.copy(n_acc, &current_norm_sq);
        }).wait();

        if (current_norm_sq < accuracy_sq) break;

        std::swap(x_curr_buf, x_next_buf);
        
        q.submit([&](sycl::handler& cgh) {
            auto n_acc = norm_buf.get_access<sycl::access::mode::write>(cgh);
            cgh.fill(n_acc, 0.0f);
        });
    }

    std::vector<float> result(n);
    q.copy(x_curr_buf.get_access<sycl::access::mode::read>(), result.data()).wait();
    return result;
}