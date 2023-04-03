

#ifndef ConvLayer_H_
#define ConvLayer_H_

#include <convolution2d.h>
#include <device.h>
#include <instanceNorm2d.h>
#include <leakyReLU.h>
#include <memory.h>
#include <reflectionPad2d.h>

#include <fstream>

namespace cpu {
namespace op {

class ConvLayer : public cpu::op::Operation {
 public:
  // ConvLayer(cpu::Device *dev, int inputChannel, int inputHeight, int
  // inputWidth, int outputChannel, int kernelSize, int stride, bool reluOn=true,
  // bool instanceOn=false);
  ConvLayer(cpu::Device* dev, int inputChannel, int inputHeight, int inputWidth,
            int outputChannel, int kernelSize, int stride, bool reluOn,
            bool instanceOn);
  virtual ~ConvLayer();
  void Forward(cpu::Memory* input, cpu::Memory* output, cpu::Memory* l1);
  void LoadWeight(std::ifstream& fp);

 private:
  cpu::op::ReflectionPad2d* reflectionPad_;
  cpu::op::Convolution2d* conv2d_;
  cpu::op::InstanceNorm2d* instance_;
  cpu::op::LeakyReLU* relu_;

  int reflectionPadding_;
  int kernelSize_;
  int stride_;
  bool reluOn_;
  bool instanceOn_;
};

}  // namespace op
}  // namespace cpu

#endif
