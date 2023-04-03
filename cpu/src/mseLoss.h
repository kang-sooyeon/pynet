#pragma once
#ifndef MSELOSS_H_
#define MSELOSS_H_

#include <cmath>
#include <iostream>

#include "device.h"
#include "memory.h"
#include "operation.h"

namespace cpu {
namespace op {

class MSELoss : public cpu::op::Operation {
 public:
  MSELoss(cpu::Device *dev, int inputChannel, int inputHeight, int inputWidth,
          int reduction);
  virtual ~MSELoss();

  void Forward(cpu::Memory *input1, cpu::Memory *input2, cpu::Memory *output);

 private:
  int reduction_;
  int size_;
};

}  // namespace op
}  // namespace cpu

#endif
