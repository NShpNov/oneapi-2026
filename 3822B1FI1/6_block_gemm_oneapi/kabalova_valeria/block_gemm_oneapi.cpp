#include "block_gemm_oneapi.h"
#include <random>
#include <iostream>
#include <iomanip>

const int BLOCKSIZE = 16;


std::vector<float> GemmBlockONEAPI(const std::vector<float> a, const std::vector<float> b, size_t size, sycl::device device){
  int block_count = size / BLOCKSIZE;
  int ssize = size * size;
  std::vector<float> c(ssize, 0.0f);
  {
    sycl::queue q(device);

    sycl::buffer<float, 1> buf_a(a.data(), sycl::range<1>(ssize));
    sycl::buffer<float, 1> buf_b(b.data(), sycl::range<1>(ssize));
    sycl::buffer<float, 1> buf_c(c.data(), sycl::range<1>(ssize));

    q.submit([&](sycl::handler& h) {
      auto acc_a = buf_a.get_access<sycl::access::mode::read>(h);
      auto acc_b = buf_b.get_access<sycl::access::mode::read>(h);
      auto acc_c = buf_c.get_access<sycl::access::mode::write>(h);
      
      h.parallel_for(sycl::range<2>(block_count, block_count), [=](sycl::id<2> idx) {
        int I = idx[0];
        int J = idx[1];
        
        for (int K = 0; K < block_count; ++K) {
          for (int i = 0; i < BLOCKSIZE; ++i) {
            for (int j = 0; j < BLOCKSIZE; ++j) {
              float sum = 0.0f;
              for (int k = 0; k < BLOCKSIZE; ++k) {
                  int a_idx = (I * BLOCKSIZE + i) * size + (K * BLOCKSIZE + k);
                  int b_idx = (K * BLOCKSIZE + k) * size + (J * BLOCKSIZE + j);
                  sum += acc_a[a_idx] * acc_b[b_idx];
              }
              int c_idx = (I * BLOCKSIZE + i) * size + (J * BLOCKSIZE + j);
              acc_c[c_idx] += sum;
            }
          }
        }
      });
    }).wait();

    sycl::host_accessor result(buf_c, sycl::read_only);
    for (int i = 0; i < ssize; ++i) {
      c[i] = result[i];
    }

  }

  return c;
}


// std::vector<float> naive(const std::vector<float> a, const std::vector<float> b, size_t size){
//   std::vector<float> c(size * size, 0.0f);
    
//   for (int i = 0; i < size; ++i) {
//       for (int j = 0; j < size; ++j) {
//           for (int k = 0; k < size; ++k) {
//               c[i * size + j] += a[i * size + k] * b[k * size + j];
//           }
//       }
//     }
    
//   return c;
// }

// int main() {
//   std::cout << std::fixed << std::setprecision(6);
//   std::mt19937 gen(42);

//   float min = 0.0f, max = 1.0f;
//   std::uniform_real_distribution<float> dist(min, max);

//   const int size = 1024;
//   std::vector<float> a(size * size);
//   std::vector<float> b(size * size);
  
//   for (int i = 0; i < size * size; ++i) {
//     a[i] = dist(gen);
//     b[i] = dist(gen);
//   }

  
  
//   std::vector<float> c_naive = naive(a, b, size);

//   // std::cout << "c_naive: \n";
//   // for (size_t i = 0; i < size; ++i){
//   //   for (size_t j = 0; j < size; ++j){
//   //     std::cout << c_naive[i*size + j] << " ";
//   //   }
//   //   std::cout << "\n";
//   // }

//   std::vector<float> c_block = GemmBlockONEAPI(a, b, size, sycl::device(sycl::default_selector_v));\

//   // std::cout << "\n";
//   // std::cout << "c_block: \n";
//   // for (size_t i = 0; i < size; ++i){
//   //   for (size_t j = 0; j < size; ++j){
//   //     std::cout << c_block[i*size + j] << " ";
//   //   }
//   //   std::cout << "\n";
//   // }
  
//   //std::cout << "\n";

//   //std::cout << std::fixed << std::setprecision(6) << a[0] << " " << b[0]<< "\n";

//   for (size_t i = 0; i < size * size; ++i) {
//       if (std::abs(c_naive[i] - c_block[i]) > 0.001) {
//         std::cout << "Error at position " << i << "\n";
//         std::cout << c_naive[i] << " " << c_block[i] << "\n";
//         break;
//       }
//   }
  
//   return 0;
// }