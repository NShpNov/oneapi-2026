#include "integral_oneapi.h"

#include <cmath>
#include <vector>
#include <algorithm>
#include <cstddef>
#include <stdexcept>
float IntegralONEAPI(float start, float end, int count, sycl::device device) {
    if (count <= 0) return 0.0f;
    if (start > end) std::swap(start, end);

    const size_t n = static_cast<size_t>(count);
    const size_t total = n * n;
    if (total == 0) return 0.0f;

    const float dx = (end - start) / static_cast<float>(count);
    const float area = dx * dx;

    sycl::queue q(device);

    const size_t max_local = 256;
    const size_t local_size = std::min(max_local, total);
    const size_t num_groups = (total + local_size - 1) / local_size;
    const size_t global = num_groups * local_size;

    std::vector<float> partials(num_groups, 0.0f);

    {
        sycl::buffer<float, 1> partial_buf(partials.data(), sycl::range<1>(partials.size()));

        q.submit([&](sycl::handler& h) {
            auto out = partial_buf.get_access<sycl::access::mode::discard_write>(h);
            sycl::local_accessor<float, 1> scratch(sycl::range<1>(local_size), h);

            h.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(local_size)),
                [=](sycl::nd_item<1> item) {
                    const size_t lid = item.get_local_id(0);
                    const size_t gid = item.get_global_id(0);
                    const size_t group = item.get_group(0);

                    float local_sum = 0.0f;

                    const size_t stride_global = item.get_global_range(0);
                    for (size_t k = gid; k < total; k += stride_global) {
                        const size_t i = k / n;
                        const size_t j = k % n;
                        const float x = start + (static_cast<float>(i) + 0.5f) * dx;
                        const float y = start + (static_cast<float>(j) + 0.5f) * dx;
                        local_sum += sycl::sin(x) * sycl::cos(y);
                    }

                    scratch[lid] = local_sum;
                    item.barrier(sycl::access::fence_space::local_space);

                    for (size_t s = item.get_local_range(0) / 2; s > 0; s >>= 1) {
                        if (lid < s) scratch[lid] += scratch[lid + s];
                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    if (lid == 0) {
                        out[group] = scratch[0];
                    }
                }
            );
        });

        q.wait_and_throw();
    }

    float sum = 0.0f;
    for (float v : partials) sum += v;

    return sum * area;
}