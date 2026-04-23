#include "block_gemm_oneapi.h"

std::vector<float> GemmBlockONEAPI(const std::vector<float>& a, const std::vector<float>& b, size_t size, sycl::device device) {
    std::vector<float> resultMatrix(size * size, 0.0f);

    const size_t blockSize = 16;

    sycl::queue q(device);

    sycl::buffer<float, 1> bufferA(a.data(), sycl::range<1>(size * size));
    sycl::buffer<float, 1> bufferB(b.data(), sycl::range<1>(size * size));
    sycl::buffer<float, 1> bufferC(resultMatrix.data(), sycl::range<1>(size * size));

    sycl::range<2> globalRange(size, size);
    sycl::range<2> localRange(blockSize, blockSize);

    q.submit([&](sycl::handler& h) {
        sycl::accessor accessorA(bufferA, h, sycl::read_only);
        sycl::accessor accessorB(bufferB, h, sycl::read_only);
        sycl::accessor accessorC(bufferC, h, sycl::write_only, sycl::no_init);

        sycl::local_accessor<float, 2> localA(sycl::range<2>(blockSize, blockSize), h);
        sycl::local_accessor<float, 2> localB(sycl::range<2>(blockSize, blockSize), h);

        h.parallel_for(sycl::nd_range<2>(globalRange, localRange), [=](sycl::nd_item<2> item) {
            size_t row = item.get_global_id(0);
            size_t col = item.get_global_id(1);
            size_t localRow = item.get_local_id(0);
            size_t localCol = item.get_local_id(1);

            float sumValue = 0.0f;

            size_t numblocks = size / blockSize;

            for (size_t t = 0; t < numblocks; ++t) {
                localA[localRow][localCol] = accessorA[row * size + (t * blockSize + localCol)];
                localB[localRow][localCol] = accessorB[(t * blockSize + localRow) * size + col];

                item.barrier(sycl::access::fence_space::local_space);

                for (size_t k = 0; k < blockSize; ++k) {
                    sumValue += localA[localRow][k] * localB[k][localCol];
                }

                item.barrier(sycl::access::fence_space::local_space);
            }

            accessorC[row * size + col] = sumValue;
            });
        });

    q.wait();

    return resultMatrix;
}