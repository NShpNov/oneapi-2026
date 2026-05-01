#include "integral_kokkos.h"

float IntegralKokkos(float start, float end, int count) {
    if (count <= 0) return 0.0f;

    const double a = static_cast<double>(start);
    const double b = static_cast<double>(end);
    const int n = count;
    const double dx = (b - a) / static_cast<double>(n);
    const double dy = dx;
    double sum = 0.0;

    using policy_type = Kokkos::MDRangePolicy<Kokkos::SYCL,Kokkos::Rank<2>>;

    policy_type policy({0, 0}, {n, n});

    Kokkos::parallel_reduce(
        "integral_midpoint",
        policy,
        KOKKOS_LAMBDA(const int i, const int j, double &local_sum) {
            const double x = a + (static_cast<double>(i) + 0.5) * dx;
            const double y = a + (static_cast<double>(j) + 0.5) * dy;
            local_sum += std::sin(x) * std::cos(y) * dx * dy;
        },
        sum);

    Kokkos::fence();

    return static_cast<float>(sum);
}