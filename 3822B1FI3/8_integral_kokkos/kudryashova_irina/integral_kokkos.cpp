#include "integral_kokkos.h"
#include <cmath>
#include <cstdint>

float IntegralKokkos(float start, float end, int count)
{
    if (count <= 0)
    {
        return 0.0f;
    }

    using execution_space = Kokkos::SYCL;

    const float step = (end - start) / static_cast<float>(count);
    const float offset = 0.5f * step;

    float sum = 0.0f;

    Kokkos::parallel_reduce(
        "double_integral_middle_riemann",
        Kokkos::MDRangePolicy<execution_space, Kokkos::Rank<2>>({0, 0}, {count, count}),
        KOKKOS_LAMBDA(const int i, const int j, float &local_sum) {
            const float x = start + static_cast<float>(i) * step + offset;
            const float y = start + static_cast<float>(j) * step + offset;

            local_sum += sinf(x) * cosf(y);
        },
        sum);

    return sum * step * step;
}
