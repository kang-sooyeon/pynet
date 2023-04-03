
#include <upsampleConvLayer.h>

namespace cpu {
namespace op {

UpsampleConvLayer::UpsampleConvLayer(cpu::Device *dev, int inputChannel,
                                     int inputHeight, int inputWidth,
                                     int outputChannel, int kernelSize,
                                     int scale, int stride, bool reluOn)
    : cpu::op::Operation(dev, inputChannel, inputHeight, inputWidth),
      kernelSize_(kernelSize),
      scale_(scale),
      stride_(stride),
      reluOn_(reluOn) {
  int h, w, c;
  // upsample layer
  upsample_ =
      new Upsample(dev, inputChannel_, inputHeight_, inputWidth_, scale_);
  upsample_->GetOutputSize(c, h, w);

  // reflectionpad
  reflectionPadding_ = kernelSize_ / 2;
  reflectionPad_ = new ReflectionPad2d(dev, c, h, w, reflectionPadding_);
  reflectionPad_->GetOutputSize(c, h, w);

  // convolution2d
  conv2d_ =
      new Convolution2d(dev, c, h, w, outputChannel, kernelSize_, stride_);
  conv2d_->GetOutputSize(c, h, w);

  // leakyReLU
  if (reluOn_) {
    relu_ = new LeakyReLU(dev, c, h, w, 0.2);
    relu_->GetOutputSize(c, h, w);
  }

  outputChannel_ = c;
  outputHeight_ = h;
  outputWidth_ = w;
}

UpsampleConvLayer::~UpsampleConvLayer() { delete conv2d_; }

void UpsampleConvLayer::LoadWeight(std::ifstream &fp) {
  conv2d_->LoadWeight(fp);
}

void UpsampleConvLayer::Forward(cpu::Memory *input, cpu::Memory *output,
                                cpu::Memory *l1) {
  upsample_->Forward(input, output);
  reflectionPad_->Forward(output, l1);
  // std::cout << "refection output size : " << output_->Size() << std::endl;
  conv2d_->Forward(l1, output);

  if (reluOn_) relu_->Forward(output, output);
}

}  // namespace op
}  // namespace cpu
