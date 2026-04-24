#include "jacobi_kokkos.h"
#include <cmath>
#include <algorithm>

std::vector<float> JacobiKokkos(
    const std::vector<float> &a,
    const std::vector<float> &b,
    float accuracy)
{

    const size_t n = b.size();
    if (n == 0)
        return {};
    if (a.size() != n * n)
        return {};

    accuracy = std::max(0.0f, accuracy);

    using ExecSpace = Kokkos::DefaultExecutionSpace;
    using MemSpace = ExecSpace::memory_space;

    Kokkos::View<float **, MemSpace> A("A", n, n);
    Kokkos::View<float *, MemSpace> B("B", n);
    Kokkos::View<float *, MemSpace> x_curr("x_curr", n);
    Kokkos::View<float *, MemSpace> x_next("x_next", n);

    Kokkos::View<float **, Kokkos::HostSpace> A_host("A_host", n, n);
    Kokkos::View<float *, Kokkos::HostSpace> B_host("B_host", n);

    for (size_t i = 0; i < n; ++i)
    {
        B_host(i) = b[i];
        for (size_t j = 0; j < n; ++j)
        {
            A_host(i, j) = a[i * n + j];
        }
    }

    Kokkos::deep_copy(A, A_host);
    Kokkos::deep_copy(B, B_host);

    Kokkos::deep_copy(x_curr, 0.0f);
    Kokkos::deep_copy(x_next, 0.0f);

    bool converged = false;

    for (int iter = 0; iter < ITERATIONS; ++iter)
    {
        float max_diff = 0.0f;
        Kokkos::parallel_reduce(
            "JacobiStep",
            Kokkos::RangePolicy<ExecSpace>(0, n),
            KOKKOS_LAMBDA(int i, float &local_max) {
                float sum = 0.0f;
                for (size_t j = 0; j < n; ++j)
                {
                    if (j != static_cast<size_t>(i))
                    {
                        sum += A(i, j) * x_curr(j);
                    }
                }
                float new_val = (B(i) - sum) / A(i, i);
                x_next(i) = new_val;
                float diff = Kokkos::fabs(new_val - x_curr(i));
                if (diff > local_max)
                    local_max = diff;
            },
            Kokkos::Max<float>(max_diff));

        if (max_diff < accuracy)
        {
            converged = true;
            break;
        }

        Kokkos::View<float *, MemSpace> tmp = x_curr;
        x_curr = x_next;
        x_next = tmp;
    }

    auto &final_view = converged ? x_curr : x_next;

    std::vector<float> result(n);
    Kokkos::View<float *, Kokkos::HostSpace> result_host("result_host", n);
    Kokkos::deep_copy(result_host, final_view);
    for (size_t i = 0; i < n; ++i)
        result[i] = result_host(i);

    return result;
}