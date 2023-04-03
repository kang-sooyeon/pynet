#pragma once
#ifndef MSELOSS_H_
#define MSELOSS_H_

#include <cmath>
#include <iostream>

#include "device.h"
#include "memory.h"
#include "operation.h"

namespace gpu {
namespace op {

class MSELoss : public gpu::op::Operation {
 public:
  MSELoss(gpu::Device *dev, int inputChannel, int inputHeight, int inputWidth,
          int reduction);
  virtual ~MSELoss();

  void Forward(gpu::Memory *input1, gpu::Memory *input2, gpu::Memory *output);

 private:
  int reduction_;
  int size_;
  cl_kernel kernel_;
};

}  // namespace op
}  // namespace gpu

#endif
