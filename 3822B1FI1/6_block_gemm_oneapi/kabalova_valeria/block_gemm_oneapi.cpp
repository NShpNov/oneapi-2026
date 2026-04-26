#include "block_gemm_oneapi.h"
#include <random>
#include <iostream>
#include <iomanip>

const int BLOCKSIZE = 16;
const int LOCALSIZE = 16;

std::vector<float> GemmBlockONEAPI(const std::vector<float>& a, const std::vector<float>& b, size_t size, sycl::device device){
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

      sycl::local_accessor<float, 2> local_a(sycl::range<2>(BLOCKSIZE, BLOCKSIZE), h);
      sycl::local_accessor<float, 2> local_b(sycl::range<2>(BLOCKSIZE, BLOCKSIZE), h);
      
      h.parallel_for(sycl::nd_range<2>(sycl::range<2>(block_count * LOCALSIZE, block_count * LOCALSIZE), sycl::range<2>(LOCALSIZE, LOCALSIZE)),
                                      [=](sycl::nd_item<2> item) {
        int I = item.get_group(0);
        int J = item.get_group(1);
        int local_i = item.get_local_id(0);
        int local_j = item.get_local_id(1);
            
        int global_i = I * BLOCKSIZE + local_i;
        int global_j = J * BLOCKSIZE + local_j;
        float sum = 0.0f;

            
        for (int K = 0; K < block_count; ++K) {
          int a_load_i = global_i;
          int a_load_j = K * BLOCKSIZE + local_j;
          local_a[local_i][local_j] = acc_a[a_load_i * size + a_load_j];
          
          int b_load_i = K * BLOCKSIZE + local_i;
          int b_load_j = global_j;
          local_b[local_i][local_j] = acc_b[b_load_i * size + b_load_j];
          
          item.barrier(sycl::access::fence_space::local_space);
          
          for (int k = 0; k < BLOCKSIZE; ++k) {
            sum += local_a[local_i][k] * local_b[k][local_j];
          }
          
          item.barrier(sycl::access::fence_space::local_space);
        }
            
        int c_idx = (I * BLOCKSIZE + local_i) * size + (J * BLOCKSIZE + local_j);
        acc_c[c_idx] = sum;
      });
    }).wait();
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