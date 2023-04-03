#ifndef SIGMOID_H_
#define SIGMOID_H_

#include <float.h>

#include <cmath>
#include <iostream>

#include "memory.h"
#include "operation.h"

namespace cpu {
namespace op {

class Sigmoid : public cpu::op::Operation {
 public:
  Sigmoid(Device *dev, int inputChannel, int inputHeight, int inputWidth);
  virtual ~Sigmoid();
  void Forward(cpu::Memory *input, cpu::Memory *output);
};

}  // namespace op
}  // namespace cpu

#endif
