#include "integral_kokkos.h"
#include <cmath>

float IntegralKokkos(float start, float end, int count) {
    if (count <= 0) return 0.0f;

    double step = (end - start) / count;
    double result = 0.0;

    Kokkos::parallel_reduce(
        "IntegralKokkos",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {count, count}),
        KOKKOS_LAMBDA(const int i, const int j, double& local_sum) {

            double x = start + (i + 0.5) * step;
            double y = start + (j + 0.5) * step;

            local_sum += sin(x) * cos(y) * step * step;
        },
        result
    );

    return static_cast<float>(result);
}