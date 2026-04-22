#include "jacobi_kokkos.h"

std::vector<float> JacobiKokkos(const std::vector<float> a, const std::vector<float> b, float accuracy){
  size_t n = b.size();

  Kokkos::View<float*> d_a("a", n * n);
  Kokkos::View<float*> d_b("b", n);
  Kokkos::View<float*> d_current_x("current_x", n);
  Kokkos::View<float*> d_new_x("new_x", n);

  Kokkos::deep_copy(d_a, Kokkos::View<const float*, Kokkos::HostSpace>(a.data(), n * n));
  Kokkos::deep_copy(d_b, Kokkos::View<const float*, Kokkos::HostSpace>(b.data(), n));
  Kokkos::deep_copy(d_current_x, 0.0f);

  size_t k = 0;
  while (k < ITERATIONS) {
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, n),
        KOKKOS_LAMBDA(int i) {
            float sigma = 0.0f;
            for (size_t j = 0; j < n; ++j) {
                if (j != i) {
                    size_t index = n * i + j;
                    sigma = std::fma(d_a(index), d_current_x(j), sigma);
                }
            }
            d_new_x(i) = (d_b(i) - sigma) / d_a(n * i + i);
        }
    );
      
    Kokkos::fence();
      
    float norm = 0.0f;
    Kokkos::parallel_reduce(Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, n),
      KOKKOS_LAMBDA(int i, float& max_norm) {
        float diff = std::abs(d_new_x(i) - d_current_x(i));
        if (diff > max_norm) max_norm = diff;
      },
      Kokkos::Max<float>(norm)
    );
      
    Kokkos::fence();
      
    if (norm < accuracy) {
      std::vector<float> result(n);
      Kokkos::deep_copy(Kokkos::View<float*, Kokkos::HostSpace>(result.data(), n), d_new_x);
      return result;
    }
      
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, n),
      KOKKOS_LAMBDA(int i) {
        d_current_x(i) = d_new_x(i);
      }
    );
      
      Kokkos::fence();
      k++;
  }
  
  std::vector<float> result(n);
  Kokkos::deep_copy(Kokkos::View<float*, Kokkos::HostSpace>(result.data(), n), d_new_x);
  return result;
}



// int main(int argc, char* argv[]) {
//   Kokkos::initialize(argc, argv);

//   size_t n = 3;
//   std::vector<float> a = {
//     4.0, 1.0, 1.0,
//     1.0, 5.0, 1.0,
//     1.0, 1.0, 6.0
//   };

//   std::vector<float> b = {6.0, 7.0, 8.0}; 
//   std::vector<float> x(n);
//   x = JacobiKokkos(a, b, 0.0001);
//   for (int i = 0; i < n; ++i) {
//     std::cout << "x" << i + 1 << " = " << std::fixed << std::setprecision(6) << x[i] << "\n";
//   }

//   Kokkos::finalize();
//   return 0;
// }