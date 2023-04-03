#pragma once

#include <float.h>
#include <memory.h>
#include <operation.h>

#include <iomanip>
#include <iostream>
#include <vector>

namespace gpu {
namespace op {

class Cat : public gpu::op::Operation {
 public:
  Cat(gpu::Device *dev, int inputChannel1, int inputHeight1, int inputWidth1,
      int inputChannel2, int inputHeight2, int inputWidth2, int dim);
  virtual ~Cat();
  void Forward(gpu::Memory *input1, gpu::Memory *input2, gpu::Memory *output);

 private:
  int inputChannel2_, inputHeight2_, inputWidth2_;
  int inputImageSize1, inputImageSize2;
  int dim_;
  cl_kernel kernel_, kernel_img;
};

}  // namespace op
}  // namespace gpu
