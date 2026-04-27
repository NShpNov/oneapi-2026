#include "block_gemm_oneapi.h"

std::vector<float> GemmBlockONEAPI(
        const std::vector<float>& a,
        const std::vector<float>& b,
        size_t size,
        sycl::device device) {

    if (size == 0 || a.size() != size * size || b.size() != size * size)
        return {};

    sycl::queue q(device);

    std::vector<float> result(size * size, 0.0f);

    constexpr size_t TILE = 16;
    size_t padded = ((size + TILE - 1) / TILE) * TILE;

    sycl::buffer<float> A(a.data(), a.size());
    sycl::buffer<float> B(b.data(), b.size());
    sycl::buffer<float> C(result.data(), result.size());

    q.submit([&](sycl::handler& h) {
        auto accA = A.get_access<sycl::access::mode::read>(h);
        auto accB = B.get_access<sycl::access::mode::read>(h);
        auto accC = C.get_access<sycl::access::mode::discard_write>(h);

        sycl::local_accessor<float, 2> localA({TILE, TILE}, h);
        sycl::local_accessor<float, 2> localB({TILE, TILE}, h);

        h.parallel_for(
            sycl::nd_range<2>({padded, padded}, {TILE, TILE}),
            [=](sycl::nd_item<2> it) {

                size_t r = it.get_global_id(0);
                size_t c = it.get_global_id(1);

                size_t lr = it.get_local_id(0);
                size_t lc = it.get_local_id(1);

                float acc = 0.0f;

                size_t blocks = (size + TILE - 1) / TILE;

                for (size_t blk = 0; blk < blocks; ++blk) {

                    size_t a_col = blk * TILE + lc;
                    size_t b_row = blk * TILE + lr;

                    localA[lr][lc] =
                        (r < size && a_col < size)
                        ? accA[r * size + a_col]
                        : 0.0f;

                    localB[lr][lc] =
                        (b_row < size && c < size)
                        ? accB[b_row * size + c]
                        : 0.0f;

                    it.barrier(sycl::access::fence_space::local_space);

                    for (size_t k = 0; k < TILE; ++k) {
                        acc += localA[lr][k] * localB[k][lc];
                    }

                    it.barrier(sycl::access::fence_space::local_space);
                }

                if (r < size && c < size) {
                    accC[r * size + c] = acc;
                }
            });
    });

    q.wait();
    return result;
}