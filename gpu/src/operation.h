

#ifndef OPERATION_H_
#define OPERATION_H_

#include <device.h>
#include <memory.h>

#include <iostream>

namespace gpu {
namespace op {

class Operation {
 public:
  Operation(Device *dev, int inputChannel, int inputHeight, int inputWidth);
  // cpu::Memory *GetOutput();
  int GetOutputSizeAll();
  void GetOutputSize(int &outputChannel, int &outputHeight, int &outputWidth);
  // void PrintOutput();
  virtual ~Operation();

 protected:
  Device *dev_;
  int inputChannel_, inputHeight_, inputWidth_;
  int outputChannel_, outputHeight_, outputWidth_;
};

}  // namespace op
}  // namespace gpu

#endif
