#include "integral_kokkos.h"

float IntegralKokkos(float start, float end, int count) {
    const float step = (end - start) / static_cast<float>(count);
    float result = 0.0f;

    Kokkos::parallel_reduce(
        "IntegralKokkos",
        Kokkos::MDRangePolicy<Kokkos::SYCL, Kokkos::Rank<2>>({0, 0}, {count, count}),
        KOKKOS_LAMBDA(const int i, const int j, float& sum) {
            const float x = start + (static_cast<float>(i) + 0.5f) * step;
            const float y = start + (static_cast<float>(j) + 0.5f) * step;
            sum += Kokkos::sin(x) * Kokkos::cos(y);
        },
        result
    );

    Kokkos::fence();

    return result * step * step;
}
