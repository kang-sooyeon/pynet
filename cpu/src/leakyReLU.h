#ifndef LEAKYRELU_H_
#define LEAKYRELU_H_

#include <float.h>

#include <iostream>

#include "memory.h"
#include "operation.h"

namespace cpu {
namespace op {

class LeakyReLU : public cpu::op::Operation {
 public:
  LeakyReLU(Device *dev, int inputChannel, int inputHeight, int inputWidth);
  LeakyReLU(Device *dev, int inputChannel, int inputHeight, int inputWidth,
            float negativeSlop);
  virtual ~LeakyReLU();
  void Forward(cpu::Memory *input, cpu::Memory *output);

 private:
  float negativeSlope_ = 0.01;
};

}  // namespace op
}  // namespace cpu

#endif
