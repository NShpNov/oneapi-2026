#include "shared_jacobi_oneapi.h"
// #include <iostream>
// #include <iomanip>


std::vector<float> JacobiSharedONEAPI(const std::vector<float>& a, const std::vector<float>& b,float accuracy, sycl::device device){
  size_t n = b.size();
  std::vector<float> result(n);
  {
    sycl::queue q(device);

    float* s_a = sycl::malloc_shared<float>(n * n, q);
    float* s_b = sycl::malloc_shared<float>(n, q);
    float* s_current_x = sycl::malloc_shared<float>(n, q);
    float* s_new_x = sycl::malloc_shared<float>(n, q);

    std::memcpy(s_a, a.data(), n * n * sizeof(float));
    std::memcpy(s_b, b.data(), n * sizeof(float));
    std::memset(s_current_x, 0, n * sizeof(float));

  size_t k = 0;
    while (k < ITERATIONS) {
      q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
        size_t i = idx[0];
        float sigma = 0.0f;
        for (size_t j = 0; j < n; ++j) {
          if (j != i) {
            size_t index = n * i + j;
            sigma = sycl::fma(s_a[index], s_current_x[j], sigma);
          }
        }
        s_new_x[i] = (s_b[i] - sigma) / s_a[n * i + i];
      }).wait();
      
      float norm = 0.0f;
      for (size_t i = 0; i < n; ++i) {
        float diff = std::abs(s_new_x[i] - s_current_x[i]);
        norm = std::max(diff, norm);
      }
      
      if (norm < accuracy) {
        std::memcpy(result.data(), s_new_x, n * sizeof(float));
        break;
      }
      
      std::memcpy(s_current_x, s_new_x, n * sizeof(float));
      
      k++;
    }
    std::memcpy(result.data(), s_new_x, n * sizeof(float));
    
    sycl::free(s_a, q);
    sycl::free(s_b, q);
    sycl::free(s_current_x, q);
    sycl::free(s_new_x, q);
  }
  
  return result;
}

// int main(){

//   size_t n = 3;
//   std::vector<float> a = {
//     4.0, 1.0, 1.0,
//     1.0, 5.0, 1.0,
//     1.0, 1.0, 6.0
//   };

//   std::vector<float> b = {6.0, 7.0, 8.0}; 
//   std::vector<float> x(n);
//   try {
//     x = JacobiSharedONEAPI(a, b, 0.0001, sycl::device(sycl::default_selector_v));
//     for (int i = 0; i < n; ++i) {
//         std::cout << "x" << i + 1 << " = " << std::fixed << std::setprecision(6) << x[i] << "\n";
//     }
//   } catch (sycl::exception const &e) {
//     std::cout << "Hello, World!";
//   }
// }