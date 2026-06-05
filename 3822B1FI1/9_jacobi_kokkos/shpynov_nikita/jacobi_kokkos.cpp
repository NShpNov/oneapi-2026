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

    Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::SYCLDeviceUSMSpace> A("A", dim, dim);
    Kokkos::View<float*, Kokkos::SYCLDeviceUSMSpace> B("B", dim);
    Kokkos::View<float*, Kokkos::SYCLDeviceUSMSpace> x("x", dim);
    Kokkos::View<float*, Kokkos::SYCLDeviceUSMSpace> x_next("x_next", dim);

    auto A_h = Kokkos::create_mirror_view(A);
    auto B_h = Kokkos::create_mirror_view(B);

    for (int i = 0; i < dim; ++i) {
        B_h(i) = b[i];
        for (int j = 0; j < dim; ++j) {
            A_h(i, j) = a[i * dim + j];
        }
    };
    Kokkos::deep_copy(A, A_h);
    Kokkos::deep_copy(B, B_h);

    Kokkos::deep_copy(x, 0.0f);
    Kokkos::deep_copy(x_next, 0.0f);
    for (int iter = 0; iter < ITERATIONS; ++iter) {
        Kokkos::parallel_for("jacobi_update", Kokkos::RangePolicy<Kokkos::SYCL>(0, dim), KOKKOS_LAMBDA(const int i) {
            float sum = 0.0f;
            for (int j = 0; j < dim; ++j) {
                if (j == i) continue;
                sum += A(i, j) * x(j);
            }

            x_next(i) = (B(i) - sum) / A(i, i);
        });

        Kokkos::fence();

        float maxdiff = 0.0;
        Kokkos::parallel_reduce("jacobi_diff", Kokkos::RangePolicy<Kokkos::SYCL>(0, dim), KOKKOS_LAMBDA(const int i, float &local_max) {
            float d = Kokkos::fabs(x_next(i) - x(i));
            if (d > local_max) local_max = d;
        }, Kokkos::Max<float>(maxdiff));

        Kokkos::fence();
        Kokkos::kokkos_swap(x, x_next);
        if (maxdiff < accuracy) {
            break;
        }

    }
    auto res_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x);
    std::vector<float> result(dim);
    for (int i = 0; i < dim; ++i) result[i] = res_h(i);
    return result;
}