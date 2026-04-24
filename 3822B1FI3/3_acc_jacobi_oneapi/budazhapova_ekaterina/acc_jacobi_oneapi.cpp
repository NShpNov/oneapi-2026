#include "acc_jacobi_oneapi.h"
#include <algorithm>
#include <cmath>

std::vector<float> JacobiAccONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device) {
    std::size_t N = static_cast<std::size_t>(std::sqrt(a.size()));
    sycl::queue q(device);
    sycl::buffer<float> buf_a(a.data(), sycl::range<1>(a.size()));
    sycl::buffer<float> buf_b(b.data(), sycl::range<1>(N));
    sycl::buffer<float> buf_x_old(sycl::range<1>(N));
    sycl::buffer<float> buf_x_new(sycl::range<1>(N));
    sycl::buffer<float> max_diff_buf(sycl::range<1>(1));

    q.submit([&](sycl::handler& h) {
        auto x_old = buf_x_old.get_access<sycl::access::mode::write>(h);
        h.parallel_for(sycl::range<1>(N),
                       [=](sycl::id<1> i) { x_old[i] = 0.0f; });
    });

    int iter = 0;
    float max_diff = 0.0f;

    for (iter = 0; iter < ITERATIONS; ++iter) {
        q.submit([&](sycl::handler& h) {
            auto A = buf_a.get_access<sycl::access::mode::read>(h);
            auto B = buf_b.get_access<sycl::access::mode::read>(h);
            auto x_old = buf_x_old.get_access<sycl::access::mode::read>(h);
            auto x_new = buf_x_new.get_access<sycl::access::mode::write>(h);
            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                std::size_t row = i[0];
                float sum = 0.0f;
                for (std::size_t j = 0; j < N; ++j) {
                    if (j != row)
                        sum += A[row * N + j] * x_old[j];
                }
                x_new[row] = (B[row] - sum) / A[row * N + row];
            });
        }).wait();

        q.submit([&](sycl::handler& h) {
            auto init = max_diff_buf.get_access<sycl::access::mode::write>(h);
            h.single_task([=]() { init[0] = 0.0f; });
        }).wait();

        q.submit([&](sycl::handler& h) {
            auto x_old = buf_x_old.get_access<sycl::access::mode::read>(h);
            auto x_new = buf_x_new.get_access<sycl::access::mode::read>(h);
            auto reduction = sycl::reduction(max_diff_buf, h,
                                             sycl::maximum<float>());
            h.parallel_for(sycl::range<1>(N), reduction,
                           [=](sycl::id<1> i, auto& max_val) {
                               float diff = sycl::fabs(x_new[i] - x_old[i]);
                               max_val = sycl::max(max_val, diff);
                           });
        }).wait();
        {
            auto acc = max_diff_buf.get_host_access();
            max_diff = acc[0];
        }

        if (max_diff < accuracy) break;
        q.submit([&](sycl::handler& h) {
            auto x_old = buf_x_old.get_access<sycl::access::mode::write>(h);
            auto x_new = buf_x_new.get_access<sycl::access::mode::read>(h);
            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                x_old[i] = x_new[i];
            });
        }).wait();
    }

    std::vector<float> result(N);
    {
        auto res_acc = buf_x_new.get_host_access();
        std::copy(res_acc.begin(), res_acc.end(), result.begin());
    }
    return result;
}