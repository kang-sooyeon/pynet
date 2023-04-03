#pragma once

#include <float.h>

#include <iomanip>
#include <iostream>
#include <vector>

#include "memory.h"
#include "operation.h"

namespace cpu {
namespace op {

class Cat : public cpu::op::Operation {
 public:
  Cat(Device *dev, int inputChannel1, int inputHeight1, int inputWidth1,
      int inputChannel2, int inputHeight2, int inputWidth2, int dim);
  virtual ~Cat();
  void Forward(cpu::Memory *input1, cpu::Memory *input2, cpu::Memory *output);

 private:
  int inputChannel2_, inputHeight2_, inputWidth2_;
  int dim_;
};

/*
class Cat3 : public cpu::op::Operation
{
public:
        Cat3(Device *dev, int inputChannel1, int inputHeight1, int inputWidth1,
int inputChannel2, int inputHeight2, int inputWidth2, int inputChannel3, int
inputHeight3, int inputWidth3, int dim);

        virtual ~Cat3();
        cpu::Memory *Forward(cpu::Memory *input1, cpu::Memory *input2);

private:
        int inputChannel3_, inputHeight3_, inputWidth3_;
        int inputChannel2_, inputHeight2_, inputWidth2_;
        int dim_;
};
*/
}  // namespace op
}  // namespace cpu
