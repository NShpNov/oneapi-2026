#include "integral_kokkos.h"
#include <Kokkos_MathematicalFunctions.hpp>

float IntegralKokkos(float start, float end, int count) {
    if (count <= 0) return 0.0f;
        bool need_finalize = false;
    if (!Kokkos::is_initialized()) {
        Kokkos::initialize();
        need_finalize = true;
    }
    const float a = start;
    const float b = end;
    const int n = count;

    const float dx = (b - a) / static_cast<float>(n);
    const float dy = dx;

    const float base = a + 0.5f * dx;

    float sum = 0.0f;

    Kokkos::parallel_reduce(
        "integral_midpoint",
        Kokkos::RangePolicy<Kokkos::SYCL>(0, n * n),
        KOKKOS_LAMBDA(const int idx, float &local_sum) {
            const int i = idx / n;
            const int j = idx % n;

            const float x = base + static_cast<float>(i) * dx;
            const float y = base + static_cast<float>(j) * dy;
            local_sum += Kokkos::sin(x) * Kokkos::cos(y);
        },
        sum);
    if (need_finalize) {
        Kokkos::finalize();
    }
    return sum * dx * dy;
}