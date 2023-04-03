#ifndef INSTANCENORM2D_H_
#define INSTANCENORM2D_H_

#include <device.h>
#include <memory.h>
#include <operation.h>

#include <cmath>
#include <fstream>
#include <iostream>

namespace cpu {
namespace op {

class InstanceNorm2d : public cpu::op::Operation {
 public:
  // InstanceNorm2d(cpu::Device *dev, int inputChannel, int inputHeight, int
  // inputWidth, bool affine = true);
  InstanceNorm2d(cpu::Device *dev, int inputChannel, int inputHeight,
                 int inputWidth, bool affine);

  virtual ~InstanceNorm2d();
  void Forward(cpu::Memory *input);
  void LoadWeight(std::ifstream &fp);
  void PrintWeight();

 private:
  float eps_;
  float momentum_;
  bool affine_;
  bool trackRunningStats_;
  int weightSize_;
  int biasSize_;

  cpu::Memory *bias_;
  cpu::Memory *weight_;
};

}  // namespace op
}  // namespace cpu

#endif
