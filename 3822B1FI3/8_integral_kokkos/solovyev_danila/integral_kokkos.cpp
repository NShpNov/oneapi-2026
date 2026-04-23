#include "integral_kokkos.h"
#include <cmath>

float IntegralKokkos(float start, float end, int count) {
    float totalSum = 0.0f;
    float stepSize = (end - start) / static_cast<float>(count);

    Kokkos::parallel_reduce(
        "IntegralCompute",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 }, { count, count }),
        KOKKOS_LAMBDA(int i, int j, float& updateSum) {
        float midX = start + (i + 0.5f) * stepSize;
        float midY = start + (j + 0.5f) * stepSize;
        updateSum += std::sin(midX) * std::cos(midY);
    },
        totalSum
    );

    return totalSum * (stepSize * stepSize);
}