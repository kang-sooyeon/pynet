

#ifndef ConvLayer_H_
#define ConvLayer_H_

#include <convolution2d.h>
#include <device.h>
#include <instanceNorm2d.h>
#include <leakyReLU.h>
#include <memory.h>
#include <reflectionPad2d.h>

#include <fstream>
#include <vector>

namespace gpu {
namespace op {

class ConvLayer : public gpu::op::Operation {
 public:
  ConvLayer(gpu::Device* dev, int inputChannel, int inputHeight, int inputWidth,
            int outputChannel, int kernelSize, int stride, bool reluOn,
            bool instanceOn, int inputN, ConvolutionMode mode,
            bool is_split = false);
  virtual ~ConvLayer();
  void Forward(gpu::Memory* input, gpu::Memory* output, gpu::Memory* l1);
  void LoadWeight(std::ifstream& fp);

 private:
  gpu::op::ReflectionPad2d* reflectionPad_;
  gpu::op::Convolution2d* conv2d_;
  gpu::op::InstanceNorm2d* instance_;
  gpu::op::LeakyReLU* relu_;
  gpu::op::ConvolutionMode mode_;

  int reflectionPadding_;
  int kernelSize_;
  int stride_;
  bool reluOn_;
  bool instanceOn_;
  int inputN_;
  bool is_split_ = false;
};

}  // namespace op
}  // namespace gpu

namespace gpu {
namespace op {

class ConvLayer2 : public gpu::op::Operation {
 public:
  ConvLayer2(gpu::Device* dev, int inputChannel, int inputHeight,
             int inputWidth, int outputChannel, int kernelSize, int stride,
             bool reluOn, bool instanceOn, int inputN, ConvolutionMode mode,
             bool is_split = false);
  virtual ~ConvLayer2();
  void Forward(std::vector<gpu::Memory*> inputs, gpu::Memory* output,
               gpu::Memory* l1, gpu::Memory* s1);
  void LoadWeight(std::ifstream& fp);
  void AddBias(gpu::Memory* bias, gpu::Memory* input);

 private:
  gpu::op::ReflectionPad2d* reflectionPad_;
  std::vector<gpu::op::Convolution2d2*> conv2d_;
  gpu::op::InstanceNorm2d* instance_;
  gpu::op::LeakyReLU* relu_;
  gpu::op::ConvolutionMode mode_;

  int reflectionPadding_;
  int kernelSize_;
  int stride_;
  bool reluOn_;
  bool instanceOn_;
  int inputN_;
  bool is_split_ = false;

  cl_kernel kernel_;
  int biasSize_;
  gpu::Memory* bias_;
};

}  // namespace op
}  // namespace gpu

#endif
