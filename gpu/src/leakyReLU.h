#ifndef LEAKYRELU_H_
#define LEAKYRELU_H_

#include <float.h>

#include <iostream>

#include "memory.h"
#include "operation.h"

namespace gpu {
namespace op {

class LeakyReLU : public gpu::op::Operation {
 public:
  LeakyReLU(Device *dev, int inputChannel, int inputHeight, int inputWidth);
  virtual ~LeakyReLU();
  void Forward(gpu::Memory *input, gpu::Memory *output);

 private:
  cl_kernel kernel_, kernel_img;
};

}  // namespace op
}  // namespace gpu

#endif
