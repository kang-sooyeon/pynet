#ifndef TRANSPOSE_H_
#define TRANSPOSE_H_

#include <float.h>

#include <cmath>
#include <iostream>

#include "memory.h"
#include "operation.h"

namespace cpu {
namespace op {

class Transpose : public cpu::op::Operation {
 public:
  Transpose(Device *dev, int inputChannel, int inputHeight, int inputWidth,
            int dim0, int dim1, int dim2);
  virtual ~Transpose();
  void Forward(cpu::Memory *input, cpu::Memory *output);

 private:
  int dim0_, dim1_, dim2_;
};

}  // namespace op
}  // namespace cpu

#endif
