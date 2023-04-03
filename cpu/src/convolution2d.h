

#ifndef CONVOLUTION2d_H_
#define CONVOLUTION2d_H_

#include <device.h>
#include <memory.h>
#include <operation.h>
#include <util.h>

#include <fstream>
#include <iostream>

namespace cpu {
namespace op {

class Convolution2d : public cpu::op::Operation {
 public:
  Convolution2d(cpu::Device *dev, int inputChannel, int inputHeight,
                int inputWidth, int outputChannel, int kernelSize, int stride);
  virtual ~Convolution2d();

  void LoadWeight(std::ifstream &fp);
  void Forward(cpu::Memory *input, cpu::Memory *output);
  void PrintWeight();

 private:
  int kernelSize_;
  int stride_;
  int weightSize_;
  int biasSize_;

  cpu::Memory *weight_;
  cpu::Memory *bias_;
};

}  // namespace op
}  // namespace cpu

#endif
