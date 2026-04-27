#include "acc_jacobi_oneapi.h"
#include <cmath>

std::vector<float> JacobiAccONEAPI(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy,
        sycl::device device) {

    size_t n = b.size();

    sycl::queue queue(device);

    std::vector<float> x_old(n, 0.0f);
    std::vector<float> x_new(n, 0.0f);

    sycl::buffer<float> a_buf(a.data(), a.size());
    sycl::buffer<float> b_buf(b.data(), b.size());
    sycl::buffer<float> x_old_buf(x_old.data(), n);
    sycl::buffer<float> x_new_buf(x_new.data(), n);

    for (int iter = 0; iter < ITERATIONS; iter++) {

        queue.submit([&](sycl::handler& h) {
            auto A = a_buf.get_access<sycl::access::mode::read>(h);
            auto B = b_buf.get_access<sycl::access::mode::read>(h);
            auto Xold = x_old_buf.get_access<sycl::access::mode::read>(h);
            auto Xnew = x_new_buf.get_access<sycl::access::mode::write>(h);

            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                float sum = 0.0f;

                for (size_t j = 0; j < n; j++) {
                    if (j != i) {
                        sum += A[i * n + j] * Xold[j];
                    }
                }

                Xnew[i] = (B[i] - sum) / A[i * n + i];
            });
        });

        float error = 0.0f;

        {
            sycl::buffer<float> err_buf(&error, 1);

            queue.submit([&](sycl::handler& h) {
                auto Xold = x_old_buf.get_access<sycl::access::mode::read>(h);
                auto Xnew = x_new_buf.get_access<sycl::access::mode::read>(h);

                auto reduction = sycl::reduction(err_buf, h, std::plus<float>());

                h.parallel_for(
                    sycl::range<1>(n),
                    reduction,
                    [=](sycl::id<1> i, auto& sum) {
                        sum += sycl::fabs(Xnew[i] - Xold[i]);
                    }
                );
            });

            queue.wait();
        }

        if (error < accuracy) {
            break;
        }

        queue.submit([&](sycl::handler& h) {
            auto Xold = x_old_buf.get_access<sycl::access::mode::read_write>(h);
            auto Xnew = x_new_buf.get_access<sycl::access::mode::read>(h);

            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                Xold[i] = Xnew[i];
            });
        });

        queue.wait();
    }

    return x_new;
}