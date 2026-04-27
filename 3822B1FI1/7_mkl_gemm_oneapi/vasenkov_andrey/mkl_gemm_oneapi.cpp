#include "mkl_gemm_oneapi.h"

#include <cstdint>
#include <oneapi/mkl.hpp>

std::vector<float> GemmMklONEAPI(
        const std::vector<float>& matrixA,
        const std::vector<float>& matrixB,
        size_t matrixSize,
        sycl::device device) {
    sycl::queue q(device, sycl::property::queue::in_order{});

    const size_t totalElements = matrixSize * matrixSize;
    const std::int64_t n = static_cast<std::int64_t>(matrixSize);
    const float alpha = 1.0f;
    const float beta = 0.0f;

    std::vector<float> resultMatrix(totalElements);

    float* d_a = sycl::aligned_alloc_device<float>(64, totalElements, q);
    float* d_b = sycl::aligned_alloc_device<float>(64, totalElements, q);
    float* d_c = sycl::aligned_alloc_device<float>(64, totalElements, q);

    q.memcpy(d_a, matrixA.data(), totalElements * sizeof(float));
    q.memcpy(d_b, matrixB.data(), totalElements * sizeof(float));

    oneapi::mkl::blas::row_major::gemm(
            q,
            oneapi::mkl::transpose::nontrans,
            oneapi::mkl::transpose::nontrans,
            n, n, n,
            alpha,
            d_a, n,
            d_b, n,
            beta,
            d_c, n);

    q.memcpy(resultMatrix.data(), d_c, totalElements * sizeof(float)).wait();

    sycl::free(d_a, q);
    sycl::free(d_b, q);
    sycl::free(d_c, q);

    return resultMatrix;
}