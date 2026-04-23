#include "jacobi_kokkos.h"

#include <algorithm>
#include <cmath>

std::vector<float> JacobiKokkos(const std::vector<float> &a, const std::vector<float> &b, float accuracy)
{
    using ExecSpace = Kokkos::SYCL;
    using MemSpace = Kokkos::SYCLDeviceUSMSpace;

    const int n = static_cast<int>(b.size());
    if (n <= 0)
    {
        return {};
    }

    Kokkos::View<float **, Kokkos::LayoutLeft, MemSpace> matrix("matrix", n, n);
    Kokkos::View<float *, MemSpace> rhs("rhs", n);
    Kokkos::View<float *, MemSpace> inv_diag("inv_diag", n);
    Kokkos::View<float *, MemSpace> prev("prev", n);
    Kokkos::View<float *, MemSpace> next("next", n);

    auto host_matrix = Kokkos::create_mirror_view(matrix);
    auto host_rhs = Kokkos::create_mirror_view(rhs);

    for (int i = 0; i < n; ++i)
    {
        host_rhs(i) = b[i];
        for (int j = 0; j < n; ++j)
        {
            host_matrix(i, j) = a[i * n + j];
        }
    }

    Kokkos::deep_copy(matrix, host_matrix);
    Kokkos::deep_copy(rhs, host_rhs);

    Kokkos::parallel_for(
        "jacobi_init", Kokkos::RangePolicy<ExecSpace>(0, n), KOKKOS_LAMBDA(const int i) {
            inv_diag(i) = 1.0f / matrix(i, i);
            prev(i) = 0.0f;
            next(i) = 0.0f;
        });

    for (int iter = 0; iter < ITERATIONS; ++iter)
    {
        Kokkos::parallel_for(
            "jacobi_iteration", Kokkos::RangePolicy<ExecSpace>(0, n), KOKKOS_LAMBDA(const int row) {
                float row_sum = 0.0f;

                for (int col = 0; col < n; ++col)
                {
                    if (col != row)
                    {
                        row_sum += matrix(row, col) * prev(col);
                    }
                }

                next(row) = (rhs(row) - row_sum) * inv_diag(row);
            });

        float max_error = 0.0f;
        Kokkos::parallel_reduce(
            "jacobi_error", Kokkos::RangePolicy<ExecSpace>(0, n),
            KOKKOS_LAMBDA(const int i, float &local_max) {
                const float diff = next(i) > prev(i) ? next(i) - prev(i) : prev(i) - next(i);

                if (diff > local_max)
                {
                    local_max = diff;
                }
            },
            Kokkos::Max<float>(max_error));

        Kokkos::kokkos_swap(prev, next);

        if (max_error < accuracy)
        {
            break;
        }
    }

    auto host_result = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), prev);

    std::vector<float> result(n);
    for (int i = 0; i < n; ++i)
    {
        result[i] = host_result(i);
    }

    return result;
}
