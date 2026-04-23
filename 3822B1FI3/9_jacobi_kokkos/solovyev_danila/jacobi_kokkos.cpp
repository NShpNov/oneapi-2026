#include "jacobi_kokkos.h"
#include <cmath>

std::vector<float> JacobiKokkos(const std::vector<float>& a, const std::vector<float>& b, float accuracy) {
    const int n = static_cast<int>(std::sqrt(a.size()));

    Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::SYCLDeviceUSMSpace> aView("aView", n, n);
    Kokkos::View<float*, Kokkos::SYCLDeviceUSMSpace> bView("bView", n);
    Kokkos::View<float*, Kokkos::SYCLDeviceUSMSpace> xOld("xOld", n);
    Kokkos::View<float*, Kokkos::SYCLDeviceUSMSpace> xNew("xNew", n);

    auto aHost = Kokkos::create_mirror_view(aView);
    auto bHost = Kokkos::create_mirror_view(bView);

    for (int i = 0; i < n; ++i) {
        bHost(i) = b[i];
        for (int j = 0; j < n; ++j) {
            aHost(i, j) = a[i * n + j];
        }
    }

    Kokkos::deep_copy(aView, aHost);
    Kokkos::deep_copy(bView, bHost);

    Kokkos::deep_copy(xOld, 0.0f);

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        Kokkos::parallel_for("jacobiKernel", n, KOKKOS_LAMBDA(int i) {
            float sigma = 0.0f;
            for (int j = 0; j < n; ++j) {
                if (i != j) {
                    sigma += aView(i, j) * xOld(j);
                }
            }
            xNew(i) = (bView(i) - sigma) / aView(i, i);
        });

        Kokkos::deep_copy(xOld, xNew);
    }

    std::vector<float> result(n);
    auto resultHost = Kokkos::create_mirror_view(xOld);
    Kokkos::deep_copy(resultHost, xOld);

    for (int i = 0; i < n; ++i) {
        result[i] = resultHost(i);
    }

    return result;
}