
#include <convMultiBlock.h>

namespace cpu {
namespace op {

ConvMultiBlock::ConvMultiBlock(cpu::Device *dev, int inputChannel,
                               int inputHeight, int inputWidth,
                               int outputChannel, int maxConvSize,
                               bool instanceOn)
    : cpu::op::Operation(dev, inputChannel, inputHeight, inputWidth),
      maxConvSize_(maxConvSize),
      instanceOn_(instanceOn) {
  int c, h, w, c2, h2, w2;

  conv3a_ =
      new cpu::op::ConvLayer(dev, inputChannel_, inputHeight_, inputWidth_,
                             outputChannel, 3, 1, true, instanceOn);
  conv3a_->GetOutputSize(c, h, w);
  conv3b_ = new cpu::op::ConvLayer(dev, outputChannel, h, w, outputChannel, 3,
                                   1, true, instanceOn);
  conv3b_->GetOutputSize(c, h, w);

  if (maxConvSize_ >= 5) {
    conv5a_ =
        new cpu::op::ConvLayer(dev, inputChannel_, inputHeight_, inputWidth_,
                               outputChannel, 5, 1, true, instanceOn);
    conv5a_->GetOutputSize(c2, h2, w2);
    conv5b_ = new cpu::op::ConvLayer(dev, outputChannel, h2, w2, outputChannel,
                                     5, 1, true, instanceOn);
    conv5b_->GetOutputSize(c2, h2, w2);
    cat5_ = new cpu::op::Cat(dev, c, h, w, c2, h2, w2, 1);
    cat5_->GetOutputSize(c, h, w);
  }

  if (maxConvSize_ >= 7) {
    conv7a_ =
        new cpu::op::ConvLayer(dev, inputChannel_, inputHeight_, inputWidth_,
                               outputChannel, 7, 1, true, instanceOn);
    conv7a_->GetOutputSize(c2, h2, w2);
    conv7b_ = new cpu::op::ConvLayer(dev, outputChannel, h2, w2, outputChannel,
                                     7, 1, true, instanceOn);
    conv7b_->GetOutputSize(c2, h2, w2);
    cat7_ = new cpu::op::Cat(dev, c, h, w, c2, h2, w2, 1);
    cat7_->GetOutputSize(c, h, w);
  }

  if (maxConvSize_ >= 9) {
    conv9a_ =
        new cpu::op::ConvLayer(dev, inputChannel_, inputHeight_, inputWidth_,
                               outputChannel, 9, 1, true, instanceOn);
    conv9a_->GetOutputSize(c2, h2, w2);
    conv9b_ = new cpu::op::ConvLayer(dev, outputChannel, h2, w2, outputChannel,
                                     9, 1, true, instanceOn);
    conv9b_->GetOutputSize(c2, h2, w2);
    cat9_ = new cpu::op::Cat(dev, c, h, w, c2, h2, w2, 1);
    cat9_->GetOutputSize(c, h, w);
  }

  outputChannel_ = c;
  outputHeight_ = h;
  outputWidth_ = w;
}

ConvMultiBlock::~ConvMultiBlock() {
  delete conv3a_;
  delete conv3b_;

  if (maxConvSize_ >= 5) {
    delete conv5a_;
    delete conv5b_;
  }
  if (maxConvSize_ >= 7) {
    delete conv7a_;
    delete conv7b_;
  }
  if (maxConvSize_ >= 9) {
    delete conv9a_;
    delete conv9b_;
  }
}

void ConvMultiBlock::LoadWeight(std::ifstream &fp) {
  conv3a_->LoadWeight(fp);
  conv3b_->LoadWeight(fp);

  if (maxConvSize_ >= 5) {
    conv5a_->LoadWeight(fp);
    conv5b_->LoadWeight(fp);
  }
  if (maxConvSize_ >= 7) {
    conv7a_->LoadWeight(fp);
    conv7b_->LoadWeight(fp);
  }
  if (maxConvSize_ >= 9) {
    conv9a_->LoadWeight(fp);
    conv9b_->LoadWeight(fp);
  }
}

void ConvMultiBlock::Forward(cpu::Memory *input, cpu::Memory *output,
                             cpu::Memory *l1, cpu::Memory *s1,
                             cpu::Memory *s2) {
  conv3a_->Forward(input, s1, l1);
  conv3b_->Forward(s1, output, l1);

  if (maxConvSize_ >= 5) {
    conv5a_->Forward(input, s1, l1);
    conv5b_->Forward(s1, s2, l1);

    cat5_->Forward(output, s2, output);
  }
  if (maxConvSize_ >= 7) {
    conv7a_->Forward(input, s1, l1);
    conv7b_->Forward(s1, s2, l1);
    cat7_->Forward(output, s2, output);
  }
  if (maxConvSize_ >= 9) {
    conv9a_->Forward(input, s1, l1);
    conv9b_->Forward(s1, s2, l1);
    cat9_->Forward(output, s2, output);
  }
}

}  // namespace op
}  // namespace cpu
