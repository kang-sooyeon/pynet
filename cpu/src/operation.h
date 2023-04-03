

#ifndef OPERATION_H_
#define OPERATION_H_

#include <iostream>

#include "device.h"
#include "memory.h"

namespace cpu {
namespace op {

class Operation {
 public:
  Operation(cpu::Device *dev, int inputChannel, int inputHeight,
            int inputWidth);
  // cpu::Memory *GetOutput();
  int GetOutputSizeAll();
  void GetOutputSize(int &outputChannel, int &outputHeight, int &outputWidth);
  // void PrintOutput();
  virtual ~Operation();

 protected:
  cpu::Device *dev_;
  int inputChannel_, inputHeight_, inputWidth_;
  int outputChannel_, outputHeight_, outputWidth_;
};

}  // namespace op
}  // namespace cpu

#endif
