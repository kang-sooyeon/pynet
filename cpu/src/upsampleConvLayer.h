

#ifndef UpsampleConvLayer_H_
#define UpsampleConvLayer_H_

#include <convolution2d.h>
#include <device.h>
#include <leakyReLU.h>
#include <memory.h>
#include <reflectionPad2d.h>
#include <upsample.h>

#include <fstream>

namespace cpu {
namespace op {

class UpsampleConvLayer : public cpu::op::Operation {
 public:
  UpsampleConvLayer(cpu::Device* dev, int inputChannel, int inputHeight,
                    int inputWidth, int outputChannel, int kernelSize,
                    int scale = 2, int stride = 1, bool reluOn = true);
  // UpsampleConvLayer(cpu::Device *dev, int inputChannel, int inputHeight, int
  // inputWidth, int outputChannel, int kernelSize, int scale, int stride, bool
  // reluOn);
  virtual ~UpsampleConvLayer();
  void LoadWeight(std::ifstream& fp);
  void Forward(cpu::Memory* input, cpu::Memory* output, cpu::Memory* l1);

 private:
  cpu::op::Upsample* upsample_;
  cpu::op::ReflectionPad2d* reflectionPad_;
  cpu::op::Convolution2d* conv2d_;
  cpu::op::LeakyReLU* relu_;

  int reflectionPadding_;
  int kernelSize_;
  int scale_;
  int stride_;
  bool reluOn_;
};

}  // namespace op
}  // namespace cpu

#endif
