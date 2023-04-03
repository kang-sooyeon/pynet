

#ifndef ConvMultiBlock_H_
#define ConvMultiBlock_H_

#include <cat.h>
#include <convLayer.h>
#include <device.h>
#include <memory.h>

#include <fstream>
#include <vector>

namespace gpu {
namespace op {

class ConvMultiBlock : public gpu::op::Operation {
 public:
  ConvMultiBlock(gpu::Device* dev, int inputChannel, int inputHeight,
                 int inputWidth, int outputChannel, int maxConvSize,
                 bool instanceOn, int inputN, bool is_split = false);
  ~ConvMultiBlock();
  void Forward(gpu::Memory* input, gpu::Memory* output, gpu::Memory* l1,
               gpu::Memory* s1);
  void LoadWeight(std::ifstream& fp);

 private:
  int maxConvSize_;
  bool instanceOn_;
  int inputN_;
  bool is_split_ = false;

  gpu::op::ConvLayer* conv3a_;
  gpu::op::ConvLayer* conv3b_;
  gpu::op::ConvLayer* conv5a_;
  gpu::op::ConvLayer* conv5b_;
  gpu::op::Cat* cat5_;
  gpu::op::ConvLayer* conv7a_;
  gpu::op::ConvLayer* conv7b_;
  gpu::op::Cat* cat7_;
  gpu::op::ConvLayer* conv9a_;
  gpu::op::ConvLayer* conv9b_;
  gpu::op::Cat* cat9_;
};

}  // namespace op
}  // namespace gpu

namespace gpu {
namespace op {

class ConvMultiBlock2 : public gpu::op::Operation {
 public:
  ConvMultiBlock2(gpu::Device* dev, int inputChannel, int inputHeight,
                  int inputWidth, int outputChannel, int maxConvSize,
                  bool instanceOn, int inputN, bool is_split = false);
  ~ConvMultiBlock2();
  void Forward(std::vector<gpu::Memory*> inputs,
               std::vector<gpu::Memory*> outputs, gpu::Memory* l1,
               gpu::Memory* s1);
  void LoadWeight(std::ifstream& fp);

 private:
  int maxConvSize_;
  bool instanceOn_;
  int inputN_;
  bool is_split_ = false;

  gpu::op::ConvLayer2* conv3a_;
  gpu::op::ConvLayer* conv3b_;
  gpu::op::ConvLayer2* conv5a_;
  gpu::op::ConvLayer* conv5b_;
  gpu::op::Cat* cat5_;
  gpu::op::ConvLayer2* conv7a_;
  gpu::op::ConvLayer* conv7b_;
  gpu::op::Cat* cat7_;
  gpu::op::ConvLayer2* conv9a_;
  gpu::op::ConvLayer* conv9b_;
  gpu::op::Cat* cat9_;
};

}  // namespace op
}  // namespace gpu

#endif
