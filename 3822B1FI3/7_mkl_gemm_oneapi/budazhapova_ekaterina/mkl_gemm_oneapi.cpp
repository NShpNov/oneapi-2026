#include "mkl_gemm_oneapi.h"
#include <oneapi/mkl.hpp>

std::vector<float> GemmMklONEAPI(
    const std::vector<float>& a, const std::vector<float>& b,
    size_t size, sycl::device device)
{
    sycl::queue q(device);
    std::vector<float> c(size * size, 0.0f);

    {
        sycl::buffer<float> a_buf(a.data(), a.size());
        sycl::buffer<float> b_buf(b.data(), b.size());
        sycl::buffer<float> c_buf(c.data(), c.size());

        oneapi::mkl::blas::row_major::gemm(
            q,
            oneapi::mkl::transpose::nontrans,
            oneapi::mkl::transpose::nontrans,
            size, size, size,
            1.0f,
            a_buf, size,
            b_buf, size,
            0.0f,
            c_buf, size
        );
    }

    q.wait();
    return c;
}