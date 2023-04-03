
#include <convLayer.h>

namespace cpu {
namespace op {

ConvLayer::ConvLayer(cpu::Device *dev, int inputChannel, int inputHeight,
                     int inputWidth, int outputChannel, int kernelSize,
                     int stride, bool reluOn, bool instanceOn)
    : cpu::op::Operation(dev, inputChannel, inputHeight, inputWidth),
      kernelSize_(kernelSize),
      stride_(stride),
      reluOn_(reluOn),
      instanceOn_(instanceOn) {
  int c, h, w;

  // reflectionpad
  reflectionPadding_ = kernelSize_ / 2;
  reflectionPad_ = new ReflectionPad2d(dev, inputChannel, inputHeight,
                                       inputWidth, reflectionPadding_);
  reflectionPad_->GetOutputSize(c, h, w);

  // convolution2d
  conv2d_ =
      new Convolution2d(dev, c, h, w, outputChannel, kernelSize_, stride_);
  conv2d_->GetOutputSize(c, h, w);

  // instanceNorm2d
  if (instanceOn_) {
    instance_ = new InstanceNorm2d(dev, c, h, w, true);
    instance_->GetOutputSize(c, h, w);
  }

  // leakyReLU
  if (reluOn_) {
    relu_ = new LeakyReLU(dev, c, h, w, 0.2);
    relu_->GetOutputSize(c, h, w);
  }

  outputChannel_ = c;
  outputHeight_ = h;
  outputWidth_ = w;
}

void ConvLayer::LoadWeight(std::ifstream &fp) {
  conv2d_->LoadWeight(fp);

  if (instanceOn_) {
    instance_->LoadWeight(fp);
  }
}

ConvLayer::~ConvLayer() {
  delete conv2d_;
  if (instanceOn_) delete instance_;
}

void ConvLayer::Forward(cpu::Memory *input, cpu::Memory *output,
                        cpu::Memory *l1) {
  // Print("input", input);
  reflectionPad_->Forward(input, l1);
  // Print("refelct", m1);
  conv2d_->Forward(l1, output);
  // Print("conv2d", output);

  if (instanceOn_) {
    instance_->Forward(output);
    // Print("instance out", output);
  }

  if (reluOn_) {
    relu_->Forward(output, output);
    // Print("relu out", output);
  }

  // Print("output", output);
  // exit(-1);
}

}  // namespace op
}  // namespace cpu
