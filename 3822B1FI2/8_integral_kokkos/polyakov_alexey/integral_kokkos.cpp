#include "integral_kokkos.h"

#include <cmath>

float IntegralKokkos(float start, float end, int count) {
    if (count <= 0) {
        return 0.0f;
    }

    const int total = count * count;
    const float step = (end - start) / count;
    const float area = step * step;

    float result = 0.0f;

    Kokkos::parallel_reduce(
        "Integral",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, total),
        KOKKOS_LAMBDA(const int idx, float& sum) {
        int i = idx / count;
        int j = idx % count;

        float x = start + (i + 0.5f) * step;
        float y = start + (j + 0.5f) * step;

        sum += Kokkos::sin(x) * Kokkos::cos(y) * area;
    },
        result
    );

    return result;
}