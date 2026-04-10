#include "jacobi_kokkos.h"
#include <Kokkos_MathematicalFunctions.hpp>

std::vector<float> JacobiKokkos(const std::vector<float> &a,
                                const std::vector<float> &b, float accuracy) {

  using ExecSpace = Kokkos::SYCL;
  using MemSpace = Kokkos::SYCLDeviceUSMSpace;
  using Layout = Kokkos::LayoutRight;

  const int n = static_cast<int>(b.size());

  Kokkos::View<float **, Layout, MemSpace> d_a("A", n, n);
  Kokkos::View<float *, MemSpace> d_b("b", n);
  Kokkos::View<float *, MemSpace> x_curr("x_curr", n);
  Kokkos::View<float *, MemSpace> x_next("x_next", n);
  Kokkos::View<float *, MemSpace> inv_d("inv_d", n);

  auto h_a = Kokkos::create_mirror_view(d_a);
  auto h_b = Kokkos::create_mirror_view(d_b);

  for (int i = 0; i < n; ++i) {
    h_b(i) = b[i];
    for (int j = 0; j < n; ++j) {
      h_a(i, j) = a[i * n + j];
    }
  }

  Kokkos::deep_copy(d_a, h_a);
  Kokkos::deep_copy(d_b, h_b);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(0, n), KOKKOS_LAMBDA(int i) {
        inv_d(i) = 1.0f / d_a(i, i);
        x_curr(i) = 0.0f;
      });

  const int check_interval = 8;

  for (int iter = 0; iter < ITERATIONS; ++iter) {

    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecSpace>(0, n), KOKKOS_LAMBDA(int i) {
          float sum = 0.0f;
          for (int j = 0; j < n; ++j) {
            if (j != i) {
              sum += d_a(i, j) * x_curr(j);
            }
          }
          x_next(i) = inv_d(i) * (d_b(i) - sum);
        });

    if ((iter + 1) % check_interval == 0) {
      float max_diff = 0.0f;
      Kokkos::parallel_reduce(
          Kokkos::RangePolicy<ExecSpace>(0, n),
          KOKKOS_LAMBDA(int i, float &local_max) {
            float diff = Kokkos::fabs(x_next(i) - x_curr(i));
            if (diff > local_max)
              local_max = diff;
          },
          Kokkos::Max<float>(max_diff));

      if (max_diff < accuracy) {
        Kokkos::kokkos_swap(x_curr, x_next);
        break;
      }
    }

    Kokkos::kokkos_swap(x_curr, x_next);
  }

  auto h_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x_curr);
  std::vector<float> result(n);
  for (int i = 0; i < n; ++i) {
    result[i] = h_x(i);
  }
  return result;
}