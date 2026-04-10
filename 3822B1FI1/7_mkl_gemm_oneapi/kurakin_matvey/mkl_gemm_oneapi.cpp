#include "mkl_gemm_oneapi.h"
#include <oneapi/mkl.hpp>

std::vector<float> GemmMklONEAPI(const std::vector<float> &matrix_a,
                                 const std::vector<float> &matrix_b,
                                 size_t size, sycl::device device) {

  sycl::queue queue(device, sycl::property::queue::in_order{});

  std::vector<float> result(size * size);

  {
    sycl::buffer buf_a(matrix_a.data(), sycl::range<1>(size * size));
    sycl::buffer buf_b(matrix_b.data(), sycl::range<1>(size * size));
    sycl::buffer buf_c(result.data(), sycl::range<1>(size * size));

    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;

    oneapi::mkl::blas::row_major::gemm(queue, oneapi::mkl::transpose::nontrans,
                                       oneapi::mkl::transpose::nontrans, size,
                                       size, size, alpha, buf_a, size, buf_b,
                                       size, beta, buf_c, size);
  }

  return result;
}