#ifndef UPSAMPLE_H_
#define UPSAMPLE_H_

#include <cmath>
#include <iostream>

#include "device.h"
#include "memory.h"
#include "operation.h"

namespace gpu {
namespace op {

class Upsample : public gpu::op::Operation {
 public:
  Upsample(gpu::Device *dev, int inputChannel, int inputHeight, int inputWidth,
           int scale);
  virtual ~Upsample();

  void Forward(gpu::Memory *input, gpu::Memory *output);

 private:
  int scale_;
  cl_kernel kernel_;
};

}  // namespace op
}  // namespace gpu

#endif
