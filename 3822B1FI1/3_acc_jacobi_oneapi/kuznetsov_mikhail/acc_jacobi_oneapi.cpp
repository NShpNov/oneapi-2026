#include "acc_jacobi_oneapi.h"

std::vector<float> JacobiAccONEAPI(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy,
        sycl::device device)
{
    const int n = b.size();
    std::vector<float> x_init(n, 0.0f);
    std::vector<float> x_tmp(n, 0.0f);

    sycl::queue q(device);

    sycl::buffer<float, 1> mat(a.data(), sycl::range<1>{a.size()});
    sycl::buffer<float, 1> vec_b(b.data(), sycl::range<1>{b.size()});
    sycl::buffer<float, 1> x_old(x_init.data(), sycl::range<1>{n});
    sycl::buffer<float, 1> x_new(x_tmp.data(), sycl::range<1>{n});

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        q.submit([&](sycl::handler& h) {
            auto A = mat.get_access<sycl::access::mode::read>(h);
            auto B = vec_b.get_access<sycl::access::mode::read>(h);
            auto Xp = x_old.get_access<sycl::access::mode::read>(h);
            auto Xc = x_new.get_access<sycl::access::mode::write>(h);

            h.parallel_for<class jacobi_step>(sycl::range<1>{n}, [=](sycl::id<1> idx) {
                int i = idx[0];
                float sum = 0.0f;

                for (int j = 0; j < n; ++j) {
                    if (j != i) {
                        sum += A[i * n + j] * Xp[j];
                    }
                }

                float diag = A[i * n + i];
                float x_val = (std::fabs(diag) < 1e-12f)
                                ? Xp[i]
                                : (B[i] - sum) / diag;

                Xc[i] = x_val;
            });
        });

        bool converged = true;
        {
            auto Xc_h = x_new.get_access<sycl::access::mode::read>();
            auto Xp_h = x_old.get_access<sycl::access::mode::read_write>();

            for (int i = 0; i < n; ++i) {
                if (std::fabs(Xc_h[i] - Xp_h[i]) >= accuracy) {
                    converged = false;
                }
                Xp_h[i] = Xc_h[i];
            }
        }

        if (converged) break;
    }

    std::vector<float> result(n);
    {
        auto out = x_old.get_access<sycl::access::mode::read>();
        for (int i = 0; i < n; ++i) {
            result[i] = out[i];
        }
    }

    return result;
}