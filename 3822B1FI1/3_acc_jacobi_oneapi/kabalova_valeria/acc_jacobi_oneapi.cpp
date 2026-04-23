#include "acc_jacobi_oneapi.h"
// #include <iostream>
// #include <iomanip>

std::vector<float> JacobiAccONEAPI(const std::vector<float>& a, const std::vector<float>& b, float accuracy, sycl::device device) {

  size_t n = b.size();
  std::vector<float> current_x(n, 0);
  std::vector<float> new_x(n, 0);
  std::vector<float> result(n);
  {
    sycl::queue q(device);

    // std::cout << "Running on: "  << q.get_device().get_info<sycl::info::device::name>()  << std::endl;

    sycl::buffer<float> a_buf(a.data(), sycl::range<1>(n * n)); 
    sycl::buffer<float> b_buf(b.data(), sycl::range<1>(n));
    sycl::buffer<float> current_x_buf(current_x.data(), sycl::range<1>(n));
    sycl::buffer<float> new_x_buf(new_x.data(), sycl::range<1>(n));
    sycl::buffer<float> norm_buf(sycl::range<1>(1));

    size_t k = 0;
    while (k < ITERATIONS) {
      q.submit([&](sycl::handler& h) {
        sycl::accessor a_acc(a_buf, h, sycl::read_only);
        sycl::accessor b_acc(b_buf, h, sycl::read_only);
        sycl::accessor current_x_acc(current_x_buf, h, sycl::read_only);
        sycl::accessor new_x_acc(new_x_buf, h, sycl::write_only, sycl::no_init);
        
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
          size_t i = idx[0];
          float sigma = 0.0f;
            
          for (size_t j = 0; j < n; ++j) {
              if (j != i) {
                  size_t index = n * i + j;
                  sigma = sycl::fma(a_acc[index], current_x_acc[j], sigma);
              }
          }
            
          new_x_acc[i] = (b_acc[i] - sigma) / a_acc[n * i + i];
        });
      }).wait();

      q.submit([&](sycl::handler& h) {
        sycl::accessor current_x_acc(current_x_buf, h, sycl::read_only);
        sycl::accessor new_x_acc(new_x_buf, h, sycl::read_only);
        sycl::accessor norm_acc(norm_buf, h, sycl::write_only, sycl::no_init);
        
        h.single_task([=]() {
          float max_diff = 0.0f;
          for (size_t i = 0; i < n; ++i) {
            float diff = sycl::fabs(new_x_acc[i] - current_x_acc[i]);
            max_diff = sycl::fmax(diff, max_diff);
          }
          norm_acc[0] = max_diff;
        });
      }).wait();

      float norm = 0.0f;
      {
        sycl::host_accessor norm_acc(norm_buf, sycl::read_only);
        norm = norm_acc[0];
      }

      if (norm < accuracy) {
        sycl::host_accessor new_x_host(new_x_buf, sycl::read_only);
        std::vector<float> result(n);
        for (size_t i = 0; i < n; ++i) {
            result[i] = new_x_host[i];
        }
        return result;
      }

      q.submit([&](sycl::handler& h) {
        sycl::accessor current_x_acc(current_x_buf, h, sycl::write_only);
        sycl::accessor new_x_acc(new_x_buf, h, sycl::read_only);
        
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
            current_x_acc[idx] = new_x_acc[idx];
        });
      }).wait();

      k++;
    }

    sycl::host_accessor new_x_host(new_x_buf, sycl::read_only);
    for (size_t i = 0; i < n; ++i) {
      result[i] = new_x_host[i];
    }
    return result;


  }
}

// std::vector<float> test(const std::vector<float> a, const std::vector<float> b, float accuracy) {
//   size_t n = b.size();
//   std::vector<float> current_x(n, 0);
//   std::vector<float> new_x(n, 0);

//   size_t k = 0;
//   while(k < ITERATIONS){
//     for (size_t i = 0; i < n; ++i){
//       float sigma = 0.0f;
//       for (size_t j = 0; j < n; ++j){
//         if (j != i){
//           size_t index = n * i + j;
//           sigma = std::fmaf(a[index], current_x[j], sigma);
//         }
//       }
//       new_x[i] = (b[i] - sigma) / a[n * i + i];
//     }
//     float norm = 0.0f;
//     for (size_t i = 0; i < n; ++i){
//       float diff = std::abs(new_x[i] - current_x[i]);
//       norm = std::max(diff, norm);
//     }
//     if (norm < accuracy) return new_x;
//     current_x = new_x;
//     k++;
//   }
//   return new_x;
// }

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
//     x = JacobiAccONEAPI(a, b, 0.0001, sycl::device(sycl::default_selector_v));
//     for (int i = 0; i < n; ++i) {
//         std::cout << "x" << i + 1 << " = " << std::fixed << std::setprecision(6) << x[i] << "\n";
//     }
//   } catch (sycl::exception const &e) {
//     std::cout << "Hello, World!";
//   }
// }