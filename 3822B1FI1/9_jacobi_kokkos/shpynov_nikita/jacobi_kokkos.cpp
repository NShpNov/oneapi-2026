#include "jacobi_kokkos.h"
#include <Kokkos_Core.hpp>
#include <cmath>


std::vector<float> JacobiKokkos(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy) {
    const size_t dim = b.size();
    if (dim == 0) return {};
    if (a.size() != dim * dim) return {};

    Kokkos::View<float*, Kokkos::DefaultExecutionSpace::memory_space> A("A", dim * dim);
    Kokkos::View<float*, Kokkos::DefaultExecutionSpace::memory_space> B("B", dim);
    Kokkos::View<float*, Kokkos::DefaultExecutionSpace::memory_space> x("x", dim);
    Kokkos::View<float*, Kokkos::DefaultExecutionSpace::memory_space> x_next("x_next", dim);

    auto A_h = Kokkos::create_mirror_view(A);
    auto B_h = Kokkos::create_mirror_view(B);
    for (size_t i = 0; i < dim * dim; ++i) A_h(i) = a[i];
    for (size_t i = 0; i < dim; ++i) B_h(i) = b[i];
    Kokkos::deep_copy(A, A_h);
    Kokkos::deep_copy(B, B_h);

    auto x_h = Kokkos::create_mirror_view(x);
    for (size_t i = 0; i < dim; ++i) x_h(i) = 0.0f;
    Kokkos::deep_copy(x, x_h);

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        Kokkos::parallel_for("jacobi_update", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, (int)dim), KOKKOS_LAMBDA(const int ii) {
            const int i = ii;
            float sum = 0.0f;
            const int base = i * (int)dim;
            for (int j = 0; j < (int)dim; ++j) {
                if (j == i) continue;
                sum += A(base + j) * x(j);
            }
            float diag = A(base + i);
            if (diag == 0.0f) {
                x_next(i) = 0.0f;
            } else {
                x_next(i) = (B(i) - sum) / diag;
            }
        });

        Kokkos::fence();

        double maxdiff = 0.0;
        Kokkos::parallel_reduce("jacobi_diff", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, (int)dim), KOKKOS_LAMBDA(const int ii, double &local_max) {
            const int i = ii;
            double d = std::fabs((double)x_next(i) - (double)x(i));
            if (d > local_max) local_max = d;
        }, Kokkos::Max<double>(maxdiff));

        Kokkos::fence();

        if (maxdiff < static_cast<double>(accuracy)) {
            auto res_h = Kokkos::create_mirror_view(x_next);
            Kokkos::deep_copy(res_h, x_next);
            std::vector<float> result(dim);
            for (size_t i = 0; i < dim; ++i) result[i] = res_h(i);
            return result;
        }

        Kokkos::deep_copy(x, x_next);
    }

    auto res_h = Kokkos::create_mirror_view(x_next);
    Kokkos::deep_copy(res_h, x_next);
    std::vector<float> result(dim);
    for (size_t i = 0; i < dim; ++i) result[i] = res_h(i);
    return result;
}