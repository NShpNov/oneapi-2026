#include "integral_kokkos.h"
#include <Kokkos_MathematicalFunctions.hpp>

float IntegralKokkos(float start, float end, int count) {
  using ExecSpace = Kokkos::SYCL;
  using MemSpace = Kokkos::SYCLDeviceUSMSpace;

  const float h = (end - start) / static_cast<float>(count);
  const float area = h * h;

  Kokkos::View<float, MemSpace> d_result("d_result");
  Kokkos::deep_copy(d_result, 0.0f);

  Kokkos::parallel_reduce(
      "double_integral",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
          Kokkos::Array<int, 2>{0, 0}, Kokkos::Array<int, 2>{count, count}),
      KOKKOS_LAMBDA(int i, int j, float &local_sum) {
        const float x = start + (static_cast<float>(i) + 0.5f) * h;
        const float y = start + (static_cast<float>(j) + 0.5f) * h;
        local_sum += Kokkos::sin(x) * Kokkos::cos(y);
      },
      d_result);

  ExecSpace().fence();

  float h_result = 0.0f;
  Kokkos::deep_copy(h_result, d_result);

  return h_result * area;
}