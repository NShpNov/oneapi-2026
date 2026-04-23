#include "acc_jacobi_oneapi.h"

#include <cmath>
#include <vector>

std::vector<float> JacobiAccONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device) {
    const std::size_t n = static_cast<std::size_t>(std::sqrt(static_cast<float>(a.size())));
    const float limit = accuracy * accuracy;

    std::vector<float> current(n, 0.0f);
    std::vector<float> next(n, 0.0f);
    std::vector<float> diff_value(1, 0.0f);

    sycl::queue queue(device);

    {
        sycl::buffer<float, 1> a_buffer(a.data(), sycl::range<1>(a.size()));
        sycl::buffer<float, 1> b_buffer(b.data(), sycl::range<1>(b.size()));
        sycl::buffer<float, 1> current_buffer(current.data(), sycl::range<1>(n));
        sycl::buffer<float, 1> next_buffer(next.data(), sycl::range<1>(n));
        sycl::buffer<float, 1> diff_buffer(diff_value.data(), sycl::range<1>(1));

        for (int iteration = 0; iteration < ITERATIONS; ++iteration) {
            queue.submit([&](sycl::handler& handler) {
                auto a_acc = a_buffer.get_access<sycl::access::mode::read>(handler);
                auto b_acc = b_buffer.get_access<sycl::access::mode::read>(handler);
                auto cur_acc = current_buffer.get_access<sycl::access::mode::read>(handler);
                auto next_acc = next_buffer.get_access<sycl::access::mode::write>(handler);

                handler.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
                    const std::size_t i = id[0];
                    const std::size_t row = i * n;
                    float sum = 0.0f;

                    for (std::size_t j = 0; j < n; ++j) {
                        if (j != i) {
                            sum += a_acc[row + j] * cur_acc[j];
                        }
                    }

                    next_acc[i] = (b_acc[i] - sum) / a_acc[row + i];
                });
            });

            queue.submit([&](sycl::handler& handler) {
                auto red = sycl::reduction(diff_buffer, handler, sycl::plus<float>());
                auto cur_acc = current_buffer.get_access<sycl::access::mode::read>(handler);
                auto next_acc = next_buffer.get_access<sycl::access::mode::read>(handler);

                handler.parallel_for(sycl::range<1>(n), red,
                                     [=](sycl::id<1> id, auto& sum) {
                                         const std::size_t i = id[0];
                                         const float d = next_acc[i] - cur_acc[i];
                                         sum.combine(d * d);
                                     });
            });

            queue.wait();

            {
                auto host_diff = diff_buffer.get_host_access();
                if (host_diff[0] < limit) {
                    auto result_acc = next_buffer.get_host_access();
                    return std::vector<float>(result_acc.begin(), result_acc.end());
                }
            }

            queue.submit([&](sycl::handler& handler) {
                auto cur_acc = current_buffer.get_access<sycl::access::mode::write>(handler);
                auto next_acc = next_buffer.get_access<sycl::access::mode::read>(handler);

                handler.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
                    const std::size_t i = id[0];
                    cur_acc[i] = next_acc[i];
                });
            });

            queue.submit([&](sycl::handler& handler) {
                auto diff_acc = diff_buffer.get_access<sycl::access::mode::write>(handler);
                handler.single_task([=]() {
                    diff_acc[0] = 0.0f;
                });
            });

            queue.wait();
        }

        queue.submit([&](sycl::handler& handler) {
            auto cur_acc = current_buffer.get_access<sycl::access::mode::write>(handler);
            auto next_acc = next_buffer.get_access<sycl::access::mode::read>(handler);

            handler.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
                const std::size_t i = id[0];
                cur_acc[i] = next_acc[i];
            });
        });

        queue.wait();

        auto result_acc = current_buffer.get_host_access();
        return std::vector<float>(result_acc.begin(), result_acc.end());
    }
}
