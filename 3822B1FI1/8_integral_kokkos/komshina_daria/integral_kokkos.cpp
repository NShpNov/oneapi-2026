#include "integral_kokkos.h"

float IntegralKokkos(float start, float end, int count) {
    if (count <= 0) {
        return 0.0f;
    }

    const float step = (end - start) / static_cast<float>(count);
    const int total = count * count;

    double sum = 0.0;

    Kokkos::parallel_reduce(
        "IntegralKokkosMiddleRiemann",
        Kokkos::RangePolicy<Kokkos::SYCL>(0, total),
        KOKKOS_LAMBDA(const int index, double& local_sum) {
            const int i = index % count;
            const int j = index / count;

            const float x = start + (static_cast<float>(i) + 0.5f) * step;
            const float y = start + (static_cast<float>(j) + 0.5f) * step;

            local_sum += Kokkos::sin(x) * Kokkos::cos(y);
        },
        sum
    );

    Kokkos::fence();

    return static_cast<float>(sum * step * step);
}
