#include "acc_jacobi_oneapi.h"
#include <algorithm>

std::vector<float> JacobiAccONEAPI(
    const std::vector<float>& matrix_a, const std::vector<float>& vector_b,
    float eps, sycl::device dev) {
    
    const size_t dim = vector_b.size();
    std::vector<float> solution(dim);
    
    try {
        sycl::queue q(dev);
        
        sycl::buffer<float, 1> buf_a(matrix_a.data(), sycl::range<1>(matrix_a.size()));
        sycl::buffer<float, 1> buf_b(vector_b.data(), sycl::range<1>(dim));
        sycl::buffer<float, 1> buf_v1(dim);
        sycl::buffer<float, 1> buf_v2(dim);
        
        float current_error = 0.0f;
        sycl::buffer<float, 1> buf_err(&current_error, 1);

        q.submit([&](sycl::handler& h) {
            auto out = buf_v1.get_access<sycl::access::mode::write>(h);
            h.fill(out, 0.0f);
        });

        int step_count = 0;
        
        auto* curr_ptr = &buf_v1;
        auto* next_ptr = &buf_v2;

        do {
            q.submit([&](sycl::handler& h) {
                auto acc_err = buf_err.get_access<sycl::access::mode::write>(h);
                h.fill(acc_err, 0.0f);
            });

            q.submit([&](sycl::handler& h) {
                auto a = buf_a.get_access<sycl::access::mode::read>(h);
                auto b = buf_b.get_access<sycl::access::mode::read>(h);
                auto x = curr_ptr->get_access<sycl::access::mode::read>(h);
                auto x_new = next_ptr->get_access<sycl::access::mode::write>(h);
                
                auto red = sycl::reduction(buf_err, h, sycl::maximum<float>());

                h.parallel_for(sycl::range<1>(dim), red, [=](sycl::id<1> id, auto& max_val) {
                    int i = id[0];
                    float sigma = 0.0f;
                    
                    for (int j = 0; j < dim; ++j) {
                        if (i != j) {
                            sigma += a[i * dim + j] * x[j];
                        }
                    }
                    float res = (b[i] - sigma) / a[i * dim + i];
                    
                    max_val.combine(sycl::fabs(res - x[i]));
                    x_new[i] = res;
                });
            });

            std::swap(curr_ptr, next_ptr);
            step_count++;

            {
                auto host_err = buf_err.get_host_access();
                current_error = host_err[0];
            }

        } while (step_count < ITERATIONS && current_error >= eps);

        auto final_access = curr_ptr->get_host_access();
        std::copy(final_access.get_pointer(), final_access.get_pointer() + dim, solution.begin());

    } catch (sycl::exception const& ex) {
        return {};
    }

    return solution;
}