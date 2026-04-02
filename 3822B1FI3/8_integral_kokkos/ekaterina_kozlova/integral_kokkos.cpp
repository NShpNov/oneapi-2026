#include "integral_kokkos.h"
#include <cmath>

float IntegralKokkos(float start, float end, int count) {
    using ExecSpace = Kokkos::SYCL;
    
    const float step_size = (end - start) / static_cast<float>(count);
    const float rect_area = step_size * step_size;

    float sin_accum = 0.0f;
    Kokkos::parallel_reduce(
        "SinIntegral",
        Kokkos::RangePolicy<ExecSpace>(0, count),
        KOKKOS_LAMBDA(const int idx, float& local_sum) {
            const float point = start + step_size * (static_cast<float>(idx) + 0.5f);
            local_sum += sinf(point);
        },
        sin_accum
    );

    float cos_accum = 0.0f;
    Kokkos::parallel_reduce(
        "CosIntegral",
        Kokkos::RangePolicy<ExecSpace>(0, count),
        KOKKOS_LAMBDA(const int idx, float& local_sum) {
            const float point = start + step_size * (static_cast<float>(idx) + 0.5f);
            local_sum += cosf(point);
        },
        cos_accum
    );
    Kokkos::fence();
    return sin_accum * cos_accum * rect_area;
}