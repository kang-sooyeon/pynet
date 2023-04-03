#ifndef MAXPOOL2d_H_
#define MAXPOOL2d_H_

#include <float.h>

#include <iostream>

#include "device.h"
#include "memory.h"
#include "operation.h"

namespace cpu {
namespace op {

class MaxPool2d : public cpu::op::Operation {
 public:
  MaxPool2d(Device *dev, int inputChannel, int inputHeight, int inputWidth,
            int kernelSize, int stride);
  virtual ~MaxPool2d();

  void Forward(cpu::Memory *input, cpu::Memory *output);

 private:
  int kernelSize_;
  int stride_;
};

}  // namespace op
}  // namespace cpu

#endif
