#include "block_gemm_oneapi.h"
#include <sycl/sycl.hpp>
#include <algorithm>
#include <cstring>

std::vector<float> GemmBlockONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        size_t size, sycl::device device) {
    if (size == 0) return {};
    if (a.size() != size * size || b.size() != size * size) return {};

    sycl::queue q(device);

    const size_t BS = 16;

    // Allocate USM shared memory and copy inputs
    float* A = static_cast<float*>(sycl::malloc_shared(sizeof(float) * size * size, q));
    float* B = static_cast<float*>(sycl::malloc_shared(sizeof(float) * size * size, q));
    float* C = static_cast<float*>(sycl::malloc_shared(sizeof(float) * size * size, q));
    if (!A || !B || !C) {
        sycl::free(A, q); sycl::free(B, q); sycl::free(C, q);
        return {};
    }

    std::memcpy(A, a.data(), sizeof(float) * size * size);
    std::memcpy(B, b.data(), sizeof(float) * size * size);
    std::fill_n(C, size * size, 0.0f);

    // Round up global sizes to multiples of BS
    size_t g0 = ((size + BS - 1) / BS) * BS;
    size_t g1 = g0;

    sycl::range<2> global_range(g0, g1);
    sycl::range<2> local_range(BS, BS);

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 2> localA(sycl::range<2>(BS, BS), h);
        sycl::local_accessor<float, 2> localB(sycl::range<2>(BS, BS), h);

        h.parallel_for(sycl::nd_range<2>(global_range, local_range), [=](sycl::nd_item<2> it) {
            const size_t gi = it.get_global_id(0);
            const size_t gj = it.get_global_id(1);
            const size_t li = it.get_local_id(0);
            const size_t lj = it.get_local_id(1);

            if (gi >= size || gj >= size) return;

            float sum = 0.0f;
            for (size_t bk = 0; bk < size; bk += BS) {
                // load tile from A: row gi, columns bk..bk+BS-1
                size_t a_col = bk + lj;
                if (a_col < size)
                    localA[li][lj] = A[gi * size + a_col];
                else
                    localA[li][lj] = 0.0f;

                // load tile from B: rows bk..bk+BS-1, column gj
                size_t b_row = bk + li;
                if (b_row < size)
                    localB[li][lj] = B[b_row * size + gj];
                else
                    localB[li][lj] = 0.0f;

                it.barrier(sycl::access::fence_space::local_space);

                const size_t K = std::min(BS, size - bk);
                for (size_t k = 0; k < K; ++k) {
                    sum += localA[li][k] * localB[k][lj];
                }

                it.barrier(sycl::access::fence_space::local_space);
            }

            C[gi * size + gj] = sum;
        });
    });

    q.wait();

    std::vector<float> result(size * size);
    std::memcpy(result.data(), C, sizeof(float) * size * size);

    sycl::free(A, q);
    sycl::free(B, q);
    sycl::free(C, q);

    return result;
}