

#ifndef ConvMultiBlock_H_
#define ConvMultiBlock_H_

#include <cat.h>
#include <convLayer.h>
#include <device.h>
#include <memory.h>

#include <fstream>

namespace cpu {
namespace op {

class ConvMultiBlock : public cpu::op::Operation {
 public:
  ConvMultiBlock(cpu::Device* dev, int inputChannel, int inputHeight,
                 int inputWidth, int outputChannel, int maxConvSize,
                 bool instanceOn);
  ~ConvMultiBlock();
  void Forward(cpu::Memory* input, cpu::Memory* output, cpu::Memory* l1,
               cpu::Memory* s1, cpu::Memory* s2);
  void LoadWeight(std::ifstream& fp);

 private:
  int maxConvSize_;
  bool instanceOn_;

  cpu::op::ConvLayer* conv3a_;
  cpu::op::ConvLayer* conv3b_;
  cpu::op::ConvLayer* conv5a_;
  cpu::op::ConvLayer* conv5b_;
  cpu::op::Cat* cat5_;
  cpu::op::ConvLayer* conv7a_;
  cpu::op::ConvLayer* conv7b_;
  cpu::op::Cat* cat7_;
  cpu::op::ConvLayer* conv9a_;
  cpu::op::ConvLayer* conv9b_;
  cpu::op::Cat* cat9_;
};

}  // namespace op
}  // namespace cpu

#endif
