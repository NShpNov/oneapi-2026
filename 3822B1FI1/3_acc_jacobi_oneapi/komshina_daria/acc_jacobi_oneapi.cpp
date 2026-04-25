#include "acc_jacobi_oneapi.h"

std::vector<float> JacobiAccONEAPI(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy,
        sycl::device device) {
    const size_t n = b.size();

    if (n == 0 || a.size() != n * n) {
        return {};
    }

    sycl::queue q(device, sycl::property::queue::in_order{});

    std::vector<float> first(n, 0.0f);
    std::vector<float> second(n, 0.0f);

    sycl::buffer<float, 1> a_buf(a.begin(), a.end());
    sycl::buffer<float, 1> b_buf(b.begin(), b.end());
    sycl::buffer<float, 1> first_buf(first.data(), sycl::range<1>(n));
    sycl::buffer<float, 1> second_buf(second.data(), sycl::range<1>(n));

    bool current_is_first = true;

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        float max_change = 0.0f;

        {
            sycl::buffer<float, 1> change_buf(&max_change, sycl::range<1>(1));

            sycl::buffer<float, 1>& old_buf =
                current_is_first ? first_buf : second_buf;

            sycl::buffer<float, 1>& new_buf =
                current_is_first ? second_buf : first_buf;

            q.submit([&](sycl::handler& h) {
                sycl::accessor A(a_buf, h, sycl::read_only);
                sycl::accessor B(b_buf, h, sycl::read_only);
                sycl::accessor old_x(old_buf, h, sycl::read_only);
                sycl::accessor new_x(new_buf, h, sycl::write_only);

                auto max_reduction = sycl::reduction(
                    change_buf,
                    h,
                    sycl::maximum<float>()
                );

                h.parallel_for(
                    sycl::range<1>(n),
                    max_reduction,
                    [=](sycl::id<1> id, auto& max_value) {
                        const size_t row = id[0];

                        float sum = 0.0f;

                        for (size_t col = 0; col < n; ++col) {
                            if (col != row) {
                                sum += A[row * n + col] * old_x[col];
                            }
                        }

                        const float next_value =
                            (B[row] - sum) / A[row * n + row];

                        new_x[row] = next_value;

                        const float difference =
                            sycl::fabs(next_value - old_x[row]);

                        max_value.combine(difference);
                    }
                );
            });

            q.wait_and_throw();
        }

        current_is_first = !current_is_first;

        if (max_change < accuracy) {
            break;
        }
    }

    std::vector<float> result(n);

    {
        sycl::buffer<float, 1>& answer_buf =
            current_is_first ? first_buf : second_buf;

        sycl::host_accessor answer(answer_buf, sycl::read_only);

        for (size_t i = 0; i < n; ++i) {
            result[i] = answer[i];
        }
    }

    return result;
}
