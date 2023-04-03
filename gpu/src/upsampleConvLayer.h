

#ifndef UpsampleConvLayer_H_
#define UpsampleConvLayer_H_

#include <convolution2d.h>
#include <device.h>
#include <leakyReLU.h>
#include <memory.h>
#include <reflectionPad2d.h>
#include <upsample.h>
#include <util.h>

#include <fstream>
#include <vector>

namespace gpu {
namespace op {

class UpsampleConvLayer : public gpu::op::Operation {
 public:
  UpsampleConvLayer(gpu::Device *dev, int inputChannel, int inputHeight,
                    int inputWidth, int outputChannel, int kernelSize,
                    ConvolutionMode mode, int scale = 2, int stride = 1,
                    bool reluOn = true);

  virtual ~UpsampleConvLayer();
  void Forward(gpu::Memory *input, gpu::Memory *output, gpu::Memory *l1,
               gpu::Memory *l2);
  void LoadWeight(std::ifstream &fp);

 private:
  gpu::op::Upsample *upsample_;
  gpu::op::ReflectionPad2d *reflectionPad_;
  gpu::op::Convolution2d *conv2d_;
  gpu::op::LeakyReLU *relu_;
  gpu::op::ConvolutionMode mode_;

  int reflectionPadding_;
  int kernelSize_;
  int scale_;
  int stride_;
  bool reluOn_;
};

}  // namespace op
}  // namespace gpu

namespace gpu {
namespace op {

class UpsampleConvLayer2 : public gpu::op::Operation {
 public:
  UpsampleConvLayer2(gpu::Device *dev, int inputChannel, int inputHeight,
                     int inputWidth, int outputChannel, int kernelSize,
                     int inputN, ConvolutionMode mode, int scale = 2,
                     int stride = 1, bool reluOn = true);
  virtual ~UpsampleConvLayer2();
  void Forward(vector<gpu::Memory *> inputs, gpu::Memory *output,
               gpu::Memory *l1, gpu::Memory *l2);
  void LoadWeight(std::ifstream &fp);
  void AddBias(gpu::Memory *bias, gpu::Memory *input);

 private:
  gpu::op::Upsample *upsample_;
  gpu::op::ReflectionPad2d *reflectionPad_;
  std::vector<gpu::op::Convolution2d2 *> conv2d_;
  gpu::op::LeakyReLU *relu_;
  gpu::op::ConvolutionMode mode_;

  int reflectionPadding_;
  int kernelSize_;
  int scale_;
  int stride_;
  bool reluOn_;
  int inputN_;

  cl_kernel kernel_;
  int biasSize_;
  gpu::Memory *bias_;
};

}  // namespace op
}  // namespace gpu

#endif
