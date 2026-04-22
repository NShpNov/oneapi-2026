#include "jacobi_kokkos.h"
#include <cmath>

std::vector<float> JacobiKokkos(
    const std::vector<float>& a,
    const std::vector<float>& b,
    float accuracy)
{
    Kokkos::initialize();

    size_t n = b.size();
    std::vector<float> x(n, 0.0f);

    Kokkos::View<float*> d_a("a", n * n);
    Kokkos::View<float*> d_b("b", n);
    Kokkos::View<float*> d_x("x", n);
    Kokkos::View<float*> d_x_new("x_new", n);

    Kokkos::View<float*>::HostMirror h_a = Kokkos::create_mirror_view(d_a);
    Kokkos::View<float*>::HostMirror h_b = Kokkos::create_mirror_view(d_b);

    for (size_t i = 0; i < n * n; ++i)
    {
        h_a(i) = a[i];
    }
    for (size_t i = 0; i < n; ++i)
    {
        h_b(i) = b[i];
    }

    Kokkos::deep_copy(d_a, h_a);
    Kokkos::deep_copy(d_b, h_b);
    Kokkos::deep_copy(d_x, 0.0f);

    for (int iter = 0; iter < ITERATIONS; ++iter)
    {
        Kokkos::parallel_for(
            "jacobi_step",
            Kokkos::RangePolicy<>(0, n),
            KOKKOS_LAMBDA(size_t i)
        {
            float sum = d_b(i);

            for (size_t j = 0; j < n; ++j)
            {
                if (i != j)
                {
                    sum -= d_a(i * n + j) * d_x(j);
                }
            }

            d_x_new(i) = sum / d_a(i * n + i);
        });

        Kokkos::fence();

        auto h_x_new = Kokkos::create_mirror_view(d_x_new);
        auto h_x = Kokkos::create_mirror_view(d_x);
        Kokkos::deep_copy(h_x_new, d_x_new);
        Kokkos::deep_copy(h_x, d_x);

        float max_diff = 0.0f;
        for (size_t i = 0; i < n; ++i)
        {
            float diff = std::fabs(h_x_new(i) - h_x(i));
            if (diff > max_diff)
            {
                max_diff = diff;
            }
            h_x(i) = h_x_new(i);
        }

        Kokkos::deep_copy(d_x, h_x);

        if (max_diff < accuracy)
        {
            break;
        }
    }

    auto h_x_final = Kokkos::create_mirror_view(d_x);
    Kokkos::deep_copy(h_x_final, d_x);

    for (size_t i = 0; i < n; ++i)
    {
        x[i] = h_x_final(i);
    }

    Kokkos::finalize();

    return x;
}