#include "jacobi_kokkos.h"
#include <Kokkos_Core.hpp>
#include <cmath>


std::vector<float> JacobiKokkos(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy) {
     const int dim = static_cast<int>(b.size());
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

    Kokkos::parallel_for("init_x", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, dim),
        KOKKOS_LAMBDA(const int i) {
            x(i) = 0.0f;
        });

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        Kokkos::parallel_for("jacobi_update", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, dim), KOKKOS_LAMBDA(const int i) {
            float sum = 0.0f;
            const int base = i * dim;
            for (int j = 0; j < dim; ++j) {
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

        float maxdiff = 0.0;
        Kokkos::parallel_reduce("jacobi_diff", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, dim), KOKKOS_LAMBDA(const int i, float &local_max) {
            float d = Kokkos::abs(x_next(i) - x(i));
            if (d > local_max) local_max = d;
        }, Kokkos::Max<float>(maxdiff));
        Kokkos::fence();
        if (maxdiff < accuracy) {
            auto res_h = Kokkos::create_mirror_view(x_next);
            Kokkos::deep_copy(res_h, x_next);
            std::vector<float> result(dim);
            for (int i = 0; i < dim; ++i) result[i] = res_h(i);
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