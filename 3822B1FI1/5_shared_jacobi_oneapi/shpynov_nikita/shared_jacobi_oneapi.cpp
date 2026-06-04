#include "shared_jacobi_oneapi.h"
constexpr size_t LOCAL_SIZE = 64;
std::vector<float> JacobiSharedONEAPI(
    const std::vector<float>& a,
    const std::vector<float>& b,
    float accuracy,
    sycl::device device)
{
    const size_t dim = b.size();
    if (dim == 0) return {};
    if (a.size() != dim * dim) return {};
    if (accuracy < 0.0f) accuracy = 0.0f;

    const float eps2 = accuracy * accuracy;

    sycl::queue q(device, sycl::property::queue::in_order{});

    float* A   = sycl::malloc_shared<float>(dim * dim, q);
    float* B   = sycl::malloc_shared<float>(dim, q);
    float* X0  = sycl::malloc_shared<float>(dim, q);
    float* X1  = sycl::malloc_shared<float>(dim, q);
    float* INV = sycl::malloc_shared<float>(dim, q);
    float* ERR = sycl::malloc_shared<float>(1, q);

    if (!A || !B || !X0 || !X1 || !INV || !ERR) {
        if (A) sycl::free(A, q);
        if (B) sycl::free(B, q);
        if (X0) sycl::free(X0, q);
        if (X1) sycl::free(X1, q);
        if (INV) sycl::free(INV, q);
        if (ERR) sycl::free(ERR, q);
        throw std::bad_alloc();
    }

    for (size_t i = 0; i < dim * dim; ++i) A[i] = a[i];
    for (size_t i = 0; i < dim; ++i) B[i] = b[i];

    for (size_t i = 0; i < dim; ++i) {
        float d = A[i * dim + i];
        INV[i] = (d != 0.0f) ? 1.0f / d : 0.0f;
        X0[i] = 0.0f;
        X1[i] = 0.0f;
    }

    q.wait();

    auto global = sycl::range<1>((dim + LOCAL_SIZE - 1) / LOCAL_SIZE * LOCAL_SIZE);

    float* x_old = X0;
    float* x_new = X1;

    for (int iter = 0; iter < ITERATIONS; ++iter) {

        q.fill(ERR, 0.0f, 1).wait();

        sycl::event e = q.submit([&](sycl::handler& h) {

            auto red = sycl::reduction(ERR, sycl::plus<float>());

            h.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(LOCAL_SIZE)),
                red,
                [=](sycl::nd_item<1> it, auto& sum)
                {
                    size_t i = it.get_global_id(0);
                    if (i >= dim) return;

                    const size_t row = i * dim;

                    float diag_inv = INV[i];

                    float sigma = 0.0f;

                    for (size_t j = 0; j < dim; ++j) {
                        sigma += A[row + j] * x_old[j];
                    }

                    float new_x = (B[i] - (sigma - A[row + i] * x_old[i])) * diag_inv;

                    x_new[i] = new_x;

                    float diff = new_x - x_old[i];
                    sum += diff * diff;
                });
        });

        e.wait();

        float err = *ERR;

        if (err < eps2) break;

        std::swap(x_old, x_new);
    }

    std::vector<float> result(dim);
    for (size_t i = 0; i < dim; ++i)
        result[i] = x_old[i];

    sycl::free(A, q);
    sycl::free(B, q);
    sycl::free(X0, q);
    sycl::free(X1, q);
    sycl::free(INV, q);
    sycl::free(ERR, q);

    return result;
}