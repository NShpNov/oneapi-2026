#include "mkl_gemm_oneapi.h"

#include <oneapi/mkl.hpp>

std::vector<float> GemmMklONEAPI(
    const std::vector<float>& a,
    const std::vector<float>& b,
    size_t size,
    sycl::device device
) {
    sycl::queue q(device);

    const int n = static_cast<int>(size);

    std::vector<float> result(n * n, 0.0f);

    float* a_dev = sycl::malloc_shared<float>(n * n, q);
    float* b_dev = sycl::malloc_shared<float>(n * n, q);
    float* c_dev = sycl::malloc_shared<float>(n * n, q);

    for (int i = 0; i < n * n; i++) {
        a_dev[i] = a[i];
        b_dev[i] = b[i];
        c_dev[i] = 0.0f;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    oneapi::mkl::blas::row_major::gemm(
        q,
        oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans,
        n, n, n,
        alpha,
        a_dev, n,
        b_dev, n,
        beta,
        c_dev, n
    );

    q.wait();

    for (int i = 0; i < n * n; i++) {
        result[i] = c_dev[i];
    }

    sycl::free(a_dev, q);
    sycl::free(b_dev, q);
    sycl::free(c_dev, q);

    return result;
}