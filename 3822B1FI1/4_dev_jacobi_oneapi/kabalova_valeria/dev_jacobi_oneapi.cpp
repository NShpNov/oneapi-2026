#include "dev_jacobi_oneapi.h"
// #include <iostream>
// #include <iomanip>

std::vector<float> JacobiDevONEAPI(const std::vector<float>& a, const std::vector<float>& b, float accuracy, sycl::device device) {
  size_t n = b.size();
  std::vector<float> result(n);

  {
    sycl::queue q(device);

    float* d_a = sycl::malloc_device<float>(n * n, q);
    float* d_b = sycl::malloc_device<float>(n, q);
    float* d_current_x = sycl::malloc_device<float>(n, q);
    float* d_new_x = sycl::malloc_device<float>(n, q);
    float* d_norm = sycl::malloc_device<float>(1, q);

    q.memcpy(d_a, a.data(), n * n * sizeof(float)).wait();
    q.memcpy(d_b, b.data(), n * sizeof(float)).wait();
    q.memset(d_current_x, 0, n * sizeof(float)).wait();
    q.memset(d_new_x, 0, n * sizeof(float)).wait();

    size_t k = 0;
    while (k < ITERATIONS) {
      q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
        size_t i = idx[0];
        float sigma = 0.0f;
        
        for (size_t j = 0; j < n; ++j) {
          if (j != i) {
              size_t index = n * i + j;
              sigma = sycl::fma(d_a[index], d_current_x[j], sigma);
          }
        }
        
        d_new_x[i] = (d_b[i] - sigma) / d_a[n * i + i];
      }).wait();

      q.submit([&](sycl::handler& h) {
        h.single_task([=]() {
          float max_diff = 0.0f;
          for (size_t i = 0; i < n; ++i) {
              float diff = sycl::fabs(d_new_x[i] - d_current_x[i]);
              max_diff = sycl::fmax(diff, max_diff);
          }
          *d_norm = max_diff;
        });
      }).wait();
      
      float norm = 0.0f;
      q.memcpy(&norm, d_norm, sizeof(float)).wait();

      if (norm < accuracy) {
        q.memcpy(result.data(), d_new_x, n * sizeof(float)).wait();
        break;
      }

      q.memcpy(d_current_x, d_new_x, n * sizeof(float)).wait();
      k++;
    }

  q.memcpy(result.data(), d_new_x, n * sizeof(float)).wait();
  sycl::free(d_a, q);
  sycl::free(d_b, q);
  sycl::free(d_current_x, q);
  sycl::free(d_new_x, q);
  sycl::free(d_norm, q);
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
//     x = JacobiDevONEAPI(a, b, 0.0001, sycl::device(sycl::default_selector_v));
//     for (int i = 0; i < n; ++i) {
//         std::cout << "x" << i + 1 << " = " << std::fixed << std::setprecision(6) << x[i] << "\n";
//     }
//   } catch (sycl::exception const &e) {
//     std::cout << "Hello, World!";
//   }
// }