#include "mkl_gemm_oneapi.h"
#include <oneapi/mkl/blas.hpp>

std::vector<float> GemmMklONEAPI(
        const std::vector<float>& a,
        const std::vector<float>& b,
        size_t size,
        sycl::device device) {

    if (size == 0 || a.size() != size * size || b.size() != size * size)
        return {};

    sycl::queue queue(device);

    std::vector<float> c(size * size, 0.0f);

    float* A = sycl::malloc_shared<float>(a.size(), queue);
    float* B = sycl::malloc_shared<float>(b.size(), queue);
    float* C = sycl::malloc_shared<float>(c.size(), queue);

    for (size_t i = 0; i < a.size(); i++) A[i] = a[i];
    for (size_t i = 0; i < b.size(); i++) B[i] = b[i];

    float alpha = 1.0f;
    float beta = 0.0f;

    oneapi::mkl::blas::column_major::gemm(
        queue,
        oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans,
        size,
        size,
        size,
        alpha,
        B, size,
        A, size,
        beta,
        C, size
    );

    queue.wait();

    for (size_t i = 0; i < c.size(); i++) {
        c[i] = C[i];
    }

    sycl::free(A, queue);
    sycl::free(B, queue);
    sycl::free(C, queue);

    return c;
}