#include "mkl_gemm_oneapi.h"

#include <cstdint>
#include <oneapi/mkl/blas.hpp>
#include <vector>

std::vector<float> GemmMklONEAPI(
        const std::vector<float>& a,
        const std::vector<float>& b,
        size_t size,
        sycl::device device) {
    const size_t matrix_size = size * size;
    if (size == 0 || a.size() != matrix_size || b.size() != matrix_size) {
        return {};
    }

    sycl::queue q(device);
    std::vector<float> c(matrix_size, 0.0f);

    const std::int64_t n = static_cast<std::int64_t>(size);

    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;

    {
        sycl::buffer<float, 1> a_buf(a.data(), sycl::range<1>(matrix_size));
        sycl::buffer<float, 1> b_buf(b.data(), sycl::range<1>(matrix_size));
        sycl::buffer<float, 1> c_buf(c.data(), sycl::range<1>(matrix_size));
        oneapi::mkl::blas::column_major::gemm(
            q,
            oneapi::mkl::transpose::nontrans,
            oneapi::mkl::transpose::nontrans,
            n,
            n,
            n,
            alpha,
            a_buf,
            n,
            b_buf,
            n,
            beta,
            c_buf,
            n
        );
    }

    q.wait_and_throw();
    return c;
}