#include "acc_jacobi_oneapi.h"

#include <cmath>

std::vector<float> JacobiAccONEAPI(
        const std::vector<float> a, const std::vector<float> b,
        float accuracy, sycl::device device) {

    const int n = static_cast<int>(b.size());

    sycl::queue queue(device);

    sycl::buffer<float> a_buf(a.data(), a.size());
    sycl::buffer<float> b_buf(b.data(), b.size());

    std::vector<float> x_init(n, 0.0f);
    sycl::buffer<float> x1_buf(x_init.data(), x_init.size());
    sycl::buffer<float> x2_buf(sycl::range<1>(n));

    sycl::buffer<float>* x_cur = &x1_buf;
    sycl::buffer<float>* x_next = &x2_buf;

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        sycl::buffer<float> diff_buf(sycl::range<1>(1));

        queue.submit([&](sycl::handler& cgh) {
            auto a_acc   = a_buf.get_access<sycl::access::mode::read>(cgh);
            auto b_acc   = b_buf.get_access<sycl::access::mode::read>(cgh);
            auto cur_acc = x_cur->get_access<sycl::access::mode::read>(cgh);
            auto nxt_acc = x_next->get_access<sycl::access::mode::write>(cgh);
            auto red     = sycl::reduction(diff_buf, cgh, sycl::maximum<float>());

            cgh.parallel_for(sycl::range<1>(n), red, [=](sycl::id<1> id, auto& max_diff) {
                int i = static_cast<int>(id[0]);
                float sum = 0.0f;

                for (int j = 0; j < n; ++j) {
                    if (j != i) {
                        sum += a_acc[i * n + j] * cur_acc[j];
                    }
                }

                float new_val = (b_acc[i] - sum) / a_acc[i * n + i];
                nxt_acc[i] = new_val;
                max_diff.combine(sycl::fabs(new_val - cur_acc[i]));
            });
        }).wait();

        float max_diff = 0.0f;
        {
            sycl::host_accessor diff_acc(diff_buf, sycl::read_only);
            max_diff = diff_acc[0];
        }

        std::swap(x_cur, x_next);

        if (max_diff < accuracy) {
            break;
        }
    }

    sycl::host_accessor result_acc(*x_cur, sycl::read_only);
    std::vector<float> result(n);
    for (int i = 0; i < n; ++i) {
        result[i] = result_acc[i];
    }
    return result;
}
