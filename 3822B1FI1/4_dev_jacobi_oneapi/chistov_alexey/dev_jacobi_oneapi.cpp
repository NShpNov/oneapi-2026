#include "dev_jacobi_oneapi.h"
#include <cmath>
#include <algorithm>

std::vector<float> JacobiAccONEAPI(const std::vector<float> a,
                                   const std::vector<float> b,
                                   float accuracy,
                                   sycl::device device) {
  const int size = b.size();
  int iteration = 0;
  float current_error = 0.0f;
  
  std::vector<float> result(size, 0.0f);
  sycl::queue queue(device);

  float *device_matrix = sycl::malloc_device<float>(a.size(), queue);
  float *device_vector = sycl::malloc_device<float>(b.size(), queue);
  float *device_current = sycl::malloc_device<float>(size, queue);
  float *device_previous = sycl::malloc_device<float>(size, queue);
  float *device_error = sycl::malloc_device<float>(1, queue);

  queue.memcpy(device_matrix, a.data(), a.size() * sizeof(float)).wait();
  queue.memcpy(device_vector, b.data(), b.size() * sizeof(float)).wait();
  queue.memset(device_current, 0, sizeof(float) * size);
  queue.memset(device_previous, 0, sizeof(float) * size);
  queue.memset(device_error, 0, sizeof(float));

  while (iteration++ < ITERATIONS) {
    auto reduction = sycl::reduction(device_error, sycl::maximum<>());

    queue.parallel_for(sycl::range<1>(size), reduction,
                       [=](sycl::id<1> idx, auto &error) {
                         const int row = idx.get(0);
                         
                         float new_value = device_vector[row];
                         for (int col = 0; col < size; col++) {
                           if (row != col) {
                             new_value -= device_matrix[row * size + col] * 
                                         device_previous[col];
                           }
                         }
                         new_value /= device_matrix[row * size + row];
                         device_current[row] = new_value;

                         float difference = sycl::fabs(new_value - 
                                                      device_previous[row]);
                         error.combine(difference);
                       });

    queue.wait();

    queue.memcpy(&current_error, device_error, sizeof(float)).wait();
    if (current_error < accuracy)
      break;
      
    queue.memset(device_error, 0, sizeof(float)).wait();
    queue.memcpy(device_previous, device_current, size * sizeof(float)).wait();
  }

  queue.memcpy(result.data(), device_current, size * sizeof(float)).wait();

  sycl::free(device_matrix, queue);
  sycl::free(device_vector, queue);
  sycl::free(device_current, queue);
  sycl::free(device_previous, queue);
  sycl::free(device_error, queue);

  return result; //
}
