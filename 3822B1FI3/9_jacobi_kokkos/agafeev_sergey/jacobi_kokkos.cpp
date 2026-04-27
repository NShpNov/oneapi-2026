#include "jacobi_kokkos.h"
#include <cmath>

std::vector<float> JacobiKokkos(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy) {

    size_t n = b.size();

    Kokkos::View<float*> x_old("x_old", n);
    Kokkos::View<float*> x_new("x_new", n);
    Kokkos::View<float*> A("A", a.size());
    Kokkos::View<float*> B("B", b.size());

    Kokkos::deep_copy(A, Kokkos::View<const float*, Kokkos::HostSpace>(a.data(), a.size()));
    Kokkos::deep_copy(B, Kokkos::View<const float*, Kokkos::HostSpace>(b.data(), b.size()));

    Kokkos::deep_copy(x_old, 0.0f);

    float error = 0.0f;

    for (int iter = 0; iter < ITERATIONS; iter++) {

        Kokkos::parallel_for(
            "JacobiStep",
            Kokkos::RangePolicy<>(0, n),
            KOKKOS_LAMBDA(const int i) {

                float sum = 0.0f;

                for (int j = 0; j < (int)n; j++) {
                    if (j != i) {
                        sum += A(i * n + j) * x_old(j);
                    }
                }

                x_new(i) = (B(i) - sum) / A(i * n + i);
            }
        );

        Kokkos::parallel_reduce(
            "JacobiError",
            Kokkos::RangePolicy<>(0, n),
            KOKKOS_LAMBDA(const int i, float& local_max) {

                float diff = fabsf(x_new(i) - x_old(i));
                if (diff > local_max) {
                    local_max = diff;
                }
            },
            Kokkos::Max<float>(error)
        );

        if (error < accuracy) {
            break;
        }

        Kokkos::deep_copy(x_old, x_new);
    }

    std::vector<float> result(n);
    auto h_result = Kokkos::create_mirror_view(x_new);
    Kokkos::deep_copy(h_result, x_new);

    for (size_t i = 0; i < n; i++) {
        result[i] = h_result(i);
    }

    return result;
}