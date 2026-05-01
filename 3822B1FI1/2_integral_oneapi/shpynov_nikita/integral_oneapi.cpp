#include "integral_oneapi.h"

#include <cmath>
#include <stdexcept>

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
    if (count <= 0) return 0.0f;

    const size_t total = static_cast<size_t>(count) * static_cast<size_t>(count);
    const float dx = (end - start) / static_cast<float>(count);
    const float area = dx * dx;

    float sum = 0.0f;

    try {
        sycl::queue q(device);

        {
            sycl::buffer<float, 1> sum_buf(sycl::range<1>(1));

            q.submit([&](sycl::handler& h) {
                auto acc = sum_buf.get_access<sycl::access::mode::read_write>(h);

                h.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
                    size_t k = id[0];
                    int i = static_cast<int>(k / static_cast<size_t>(count));
                    int j = static_cast<int>(k % static_cast<size_t>(count));
                    float x = start + (i + 0.5f) * dx;
                    float y = start + (j + 0.5f) * dx;
                    float val = sycl::sin(x) * sycl::cos(y);
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                      sycl::memory_scope::device,
                                      sycl::access::address_space::global_space>
                        a(acc[0]);
                    a.fetch_add(val);
                });
            });

            q.wait();
            auto host_acc = sum_buf.get_access<sycl::access::mode::read>();
            sum = host_acc[0];
        }
    } catch (const std::exception&) {
        sum = 0.0f;
        for (int i = 0; i < count; ++i) {
            for (int j = 0; j < count; ++j) {
                float x = start + (i + 0.5f) * dx;
                float y = start + (j + 0.5f) * dx;
                sum += std::sin(x) * std::cos(y);
            }
        }
    }

    return sum * area;
}
