#include "mkl_gemm_oneapi.h"
#include <oneapi/mkl/blas.hpp>
#include <vector>

std::vector<float> GemmMklONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        size_t size, sycl::device device) {

    sycl::queue q(device);
    std::int64_t n = static_cast<std::int64_t>(size);
    size_t total = size * size;
    sycl::buffer<float, 1> buf_a(a.data(), sycl::range<1>(total));
    sycl::buffer<float, 1> buf_b(b.data(), sycl::range<1>(total));
    sycl::buffer<float, 1> buf_c(sycl::range<1>(total));

    float alpha = 1.0f, beta = 0.0f;
    auto trans = oneapi::mkl::transpose::nontrans;
    oneapi::mkl::blas::row_major::gemm(
        q, trans, trans, n, n, n,
        alpha, buf_a, n,
        buf_b, n,
        beta, buf_c, n);
    q.wait();
    auto host_c = buf_c.get_host_access();
    std::vector<float> result(host_c.begin(), host_c.end());
    return result;
}