#include "mkl_gemm_oneapi.h"

#include <vector>
#include <stdexcept>
#include <cstdint>

#include <oneapi/mkl.hpp>

std::vector<float> GemmMklONEAPI(
		const std::vector<float>& a, const std::vector<float>& b,
		size_t size, sycl::device device) {
	if (size == 0) return {};
	if (a.size() != size * size || b.size() != size * size) return {};

	sycl::queue q(device);

	const size_t N = size;
	const size_t nn = N * N;

	float *dev_a = sycl::malloc_device<float>(nn, q);
	float *dev_b = sycl::malloc_device<float>(nn, q);
	float *dev_c = sycl::malloc_device<float>(nn, q);
	if (!dev_a || !dev_b || !dev_c) {
		sycl::free(dev_a, q);
		sycl::free(dev_b, q);
		sycl::free(dev_c, q);
		throw std::bad_alloc();
	}

	q.memcpy(dev_a, a.data(), sizeof(float) * nn).wait();
	q.memcpy(dev_b, b.data(), sizeof(float) * nn).wait();

	const std::int64_t N64 = static_cast<std::int64_t>(N);


	auto ev = oneapi::mkl::blas::row_major::gemm(q,
			oneapi::mkl::transpose::nontrans,
			oneapi::mkl::transpose::nontrans,
			N64, N64, N64,
			1.0f,
			dev_a, N64,
			dev_b, N64,
			0.0f,
			dev_c, N64);

	ev.wait();

	std::vector<float> result(nn);
	q.memcpy(result.data(), dev_c, sizeof(float) * nn).wait();

	sycl::free(dev_a, q);
	sycl::free(dev_b, q);
	sycl::free(dev_c, q);

	return result;
}
