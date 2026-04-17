#include "block_gemm_oneapi.h"

std::vector<float> GemmBlockONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        size_t size, sycl::device device) {
    constexpr size_t TILE = 16;

    // Pad size up to the nearest multiple of TILE
    const size_t padded = ((size + TILE - 1) / TILE) * TILE;
    const size_t total = padded * padded;

    std::vector<float> result(size * size, 0.0f);

    sycl::queue q(device);

    float* d_a = sycl::malloc_device<float>(total, q);
    float* d_b = sycl::malloc_device<float>(total, q);
    float* d_c = sycl::malloc_device<float>(total, q);

    // Zero-fill padded device buffers
    q.memset(d_a, 0, total * sizeof(float));
    q.memset(d_b, 0, total * sizeof(float));
    q.memset(d_c, 0, total * sizeof(float)).wait();

    // Copy rows from host to padded device matrices
    for (size_t row = 0; row < size; ++row) {
        q.memcpy(d_a + row * padded, a.data() + row * size, size * sizeof(float));
        q.memcpy(d_b + row * padded, b.data() + row * size, size * sizeof(float));
    }
    q.wait();

    const size_t num_tiles = padded / TILE;

    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 2> tile_a({TILE, TILE}, cgh);
        sycl::local_accessor<float, 2> tile_b({TILE, TILE}, cgh);

        cgh.parallel_for(
            sycl::nd_range<2>({padded, padded}, {TILE, TILE}),
            [=](sycl::nd_item<2> item) {
                const size_t row = item.get_global_id(0);
                const size_t col = item.get_global_id(1);
                const size_t li  = item.get_local_id(0);
                const size_t lj  = item.get_local_id(1);

                float sum = 0.0f;

                for (size_t t = 0; t < num_tiles; ++t) {
                    // Load tiles collaboratively into local memory
                    tile_a[li][lj] = d_a[row * padded + t * TILE + lj];
                    tile_b[li][lj] = d_b[(t * TILE + li) * padded + col];

                    item.barrier(sycl::access::fence_space::local_space);

                    for (size_t k = 0; k < TILE; ++k) {
                        sum += tile_a[li][k] * tile_b[k][lj];
                    }

                    item.barrier(sycl::access::fence_space::local_space);
                }

                d_c[row * padded + col] = sum;
            });
    }).wait();

    // Copy result back, stripping padding
    for (size_t row = 0; row < size; ++row) {
        q.memcpy(result.data() + row * size, d_c + row * padded, size * sizeof(float));
    }
    q.wait();

    sycl::free(d_a, q);
    sycl::free(d_b, q);
    sycl::free(d_c, q);

    return result;
}
