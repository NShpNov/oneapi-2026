#include "integral_kokkos.h"
#include <iostream> 

float IntegralKokkos(float start, float end, int count){
  float result = 0.0f;
  float delta = (end - start) / count;
  
  Kokkos::View<float, Kokkos::HostSpace> result_view("result");
  result_view() = 0.0f;
  
  Kokkos::parallel_reduce(
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {count, count}),
    KOKKOS_LAMBDA(int i, int j, float& sum) {
        float x = start + (i + 0.5f) * delta;
        float y = start + (j + 0.5f) * delta;
        sum += Kokkos::sin(x) * Kokkos::cos(y);
    },
    result_view
  );
  
  Kokkos::fence();
  
  result = result_view();
  return result * delta * delta;
}

// int main(int argc, char* argv[]) {
//     Kokkos::initialize(argc, argv);
//     float start = 0.0f;
//     float end = 1.0f;
//     int count = 1000;
    
//     float result = IntegralKokkos(start, end, count);
    
//     std::cout << "Result: " << result << std::endl;
    
//     Kokkos::finalize();
//     return 0;
// }

