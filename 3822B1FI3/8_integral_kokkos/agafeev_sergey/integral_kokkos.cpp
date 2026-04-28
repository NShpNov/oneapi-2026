#include "integral_kokkos.h"
#include <cmath>
#include <cstdint>

float IntegralKokkos(float start, float end, int count) {
    if (count <= 0 || start == end) {
        return 0.0f;
    }
    using ExecSpace = Kokkos::SYCL;
    const float step = (end - start) / static_cast<float>(count);
    const float cell_area = step * step;
    const std::int64_t total_cells =
        static_cast<std::int64_t>(count) * static_cast<std::int64_t>(count);
    float result = 0.0f;
    Kokkos::parallel_reduce(
        "IntegralKokkos",
        Kokkos::RangePolicy<ExecSpace>(0, total_cells),
        KOKKOS_LAMBDA(const std::int64_t idx, float& local_sum) {
            const int j = static_cast<int>(idx / count);
            const int i = static_cast<int>(idx % count);

            const float x = start + (static_cast<float>(i) + 0.5f) * step;
            const float y = start + (static_cast<float>(j) + 0.5f) * step;

            local_sum += Kokkos::sin(x) * Kokkos::cos(y) * cell_area;
        },
        result);
    return result;
}