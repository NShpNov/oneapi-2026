#include "mkl_gemm_oneapi.h"
#include <oneapi/mkl.hpp>
#include <sycl/sycl.hpp>
#include <vector>

std::vector<float> GemmMklONEAPI(
    const std::vector<float>& a, const std::vector<float>& b,
    size_t size, sycl::device device) {

    sycl::queue deviceQueue(device);

    size_t matrixElements = size * size;
    size_t bytes = matrixElements * sizeof(float);

    float* deviceA = sycl::malloc_device<float>(matrixElements, deviceQueue);
    float* deviceB = sycl::malloc_device<float>(matrixElements, deviceQueue);
    float* deviceC = sycl::malloc_device<float>(matrixElements, deviceQueue);

    sycl::event copyEventA = deviceQueue.memcpy(deviceA, a.data(), bytes);
    sycl::event copyEventB = deviceQueue.memcpy(deviceB, b.data(), bytes);

    float alpha = 1.0f;
    float beta = 0.0f;

    std::vector<sycl::event> dependencies = { copyEventA, copyEventB };

    sycl::event gemmEvent = oneapi::mkl::blas::row_major::gemm(
        deviceQueue,
        oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans,
        size, size, size,
        alpha,
        deviceA, size,
        deviceB, size,
        beta,
        deviceC, size,
        dependencies
    );

    std::vector<float> resultVector(matrixElements);
    deviceQueue.memcpy(resultVector.data(), deviceC, bytes, { gemmEvent }).wait();

    sycl::free(deviceA, deviceQueue);
    sycl::free(deviceB, deviceQueue);
    sycl::free(deviceC, deviceQueue);

    return resultVector;
}