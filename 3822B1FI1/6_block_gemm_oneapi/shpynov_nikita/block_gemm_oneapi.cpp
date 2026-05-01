#include "block_gemm_oneapi.h"

std::vector<float> GemmBlockONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        size_t size, sycl::device device) {
    if (size == 0) return {};
	if (a.size() != size * size || b.size() != size * size) return {};
    sycl::queue q(device);

    size_t BS = 16;

    sycl::buffer<float, 1> bufA(a.data(), sycl::range<1>(size * size));
    sycl::buffer<float, 1> bufB(b.data(), sycl::range<1>(size * size));
    sycl::buffer<float, 1> bufC(sycl::range<1>(size * size));
    {
        auto acc_c = bufC.get_access<sycl::access::mode::write>();
        for (size_t i = 0; i < size * size; ++i) acc_c[i] = 0.0f;
    }
    for (size_t bi = 0; bi < size; bi += BS) {
        for (size_t bj = 0; bj < size; bj += BS) {
            for (size_t bk = 0; bk < size; bk += BS) {
                q.submit([&](sycl::handler &h) {
                    auto acc_a = bufA.get_access<sycl::access::mode::read>(h);
                    auto acc_b = bufB.get_access<sycl::access::mode::read>(h);
                    auto acc_c = bufC.get_access<sycl::access::mode::read_write>(h);

                    h.parallel_for(sycl::range<2>(BS, BS), [=](sycl::id<2> idx) {
                        const size_t i_local = idx[0];
                        const size_t j_local = idx[1];
                        const size_t i = bi + i_local;
                        const size_t j = bj + j_local;

                        float sum = 0.0f;
                        const size_t a_row = i * size;
                        const size_t b_col = j;
                        for (size_t k = 0; k < BS; ++k) {
                            const size_t ka = bk + k;
                            sum += acc_a[a_row + ka] * acc_b[ka * size + b_col];
                        }

                        auto ptr = &acc_c[a_row + j];
                        sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            atomic(*ptr);
                        atomic.fetch_add(sum);
                    });
                });
            }
        }
    }
    q.wait();
    std::vector<float> result(size * size);
    {
        auto accC_h = bufC.get_access<sycl::access::mode::read>();
        for (size_t i = 0; i < size * size; ++i) result[i] = accC_h[i];
    }
    return result;
}