#ifndef SIGMOID_H_
#define SIGMOID_H_

#include <float.h>

#include <cmath>
#include <iostream>

#include "memory.h"
#include "operation.h"

namespace gpu {
namespace op {

class Sigmoid : public gpu::op::Operation {
 public:
  Sigmoid(Device *dev, int inputChannel, int inputHeight, int inputWidth);
  virtual ~Sigmoid();
  void Forward(gpu::Memory *input, gpu::Memory *output);

 private:
  cl_kernel kernel_, kernel_img;
};

}  // namespace op
}  // namespace gpu

#endif
