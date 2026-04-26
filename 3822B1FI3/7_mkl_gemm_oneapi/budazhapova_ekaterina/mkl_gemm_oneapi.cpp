#include "mkl_gemm_oneapi.h"
#include <oneapi/mkl.hpp>
#include <vector>

std::vector<float> GemmMklONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        size_t size, sycl::device device) {

    sycl::queue q(device);
    std::int64_t n = static_cast<std::int64_t>(size);
    size_t total = size * size;

    std::vector<float> a_col(total), b_col(total);
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            a_col[j * size + i] = a[i * size + j];
            b_col[j * size + i] = b[i * size + j];
        }
    }

    sycl::buffer<float, 1> buf_a(a_col.data(), sycl::range<1>(total));
    sycl::buffer<float, 1> buf_b(b_col.data(), sycl::range<1>(total));
    sycl::buffer<float, 1> buf_c(sycl::range<1>(total));

    float alpha = 1.0f, beta = 0.0f;
    auto trans = oneapi::mkl::transpose::nontrans;
    oneapi::mkl::blas::column_major::gemm(
        q, trans, trans, n, n, n,
        alpha, buf_b, n,
        buf_a, n,
        beta, buf_c, n);
    q.wait();
    std::vector<float> result(total);
    auto host_c = buf_c.get_host_access();
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            result[i * size + j] = host_c[j * size + i];
        }
    }

    return result;
}