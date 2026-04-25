#include "shared_jacobi_oneapi.h"

std::vector<float> JacobiSharedONEAPI(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy,
        sycl::device device) {
    const size_t n = b.size();

    if (n == 0 || a.size() != n * n) {
        return {};
    }

    sycl::queue q(device, sycl::property::queue::in_order{});

    float* A = sycl::malloc_shared<float>(a.size(), q);
    float* B = sycl::malloc_shared<float>(n, q);
    float* old_x = sycl::malloc_shared<float>(n, q);
    float* new_x = sycl::malloc_shared<float>(n, q);
    float* max_diff = sycl::malloc_shared<float>(1, q);

    if (!A || !B || !old_x || !new_x || !max_diff) {
        sycl::free(A, q);
        sycl::free(B, q);
        sycl::free(old_x, q);
        sycl::free(new_x, q);
        sycl::free(max_diff, q);
        return {};
    }

    for (size_t i = 0; i < a.size(); ++i) {
        A[i] = a[i];
    }

    for (size_t i = 0; i < n; ++i) {
        B[i] = b[i];
        old_x[i] = 0.0f;
        new_x[i] = 0.0f;
    }

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        *max_diff = 0.0f;

        q.submit([&](sycl::handler& h) {
            auto diff_reduction = sycl::reduction(
                max_diff,
                0.0f,
                sycl::maximum<float>()
            );

            h.parallel_for(
                sycl::range<1>(n),
                diff_reduction,
                [=](sycl::id<1> id, auto& diff) {
                    const size_t row = id[0];

                    float sum = 0.0f;

                    for (size_t col = 0; col < n; ++col) {
                        if (col != row) {
                            sum += A[row * n + col] * old_x[col];
                        }
                    }

                    const float value = (B[row] - sum) / A[row * n + row];

                    new_x[row] = value;
                    diff.combine(sycl::fabs(value - old_x[row]));
                }
            );
        }).wait_and_throw();

        float* temp = old_x;
        old_x = new_x;
        new_x = temp;

        if (*max_diff < accuracy) {
            break;
        }
    }

    std::vector<float> result(n);

    for (size_t i = 0; i < n; ++i) {
        result[i] = old_x[i];
    }

    sycl::free(A, q);
    sycl::free(B, q);
    sycl::free(old_x, q);
    sycl::free(new_x, q);
    sycl::free(max_diff, q);

    return result;
}
