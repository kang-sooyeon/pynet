#ifndef UPSAMPLE_H_
#define UPSAMPLE_H_

#include <cmath>
#include <iostream>

#include "device.h"
#include "memory.h"
#include "operation.h"

namespace cpu {
namespace op {

class Upsample : public cpu::op::Operation {
 public:
  Upsample(cpu::Device *dev, int inputChannel, int inputHeight, int inputWidth,
           int scale);
  virtual ~Upsample();

  void Forward(cpu::Memory *input, cpu::Memory *output);

 private:
  int scale_;
};

}  // namespace op
}  // namespace cpu

#endif
