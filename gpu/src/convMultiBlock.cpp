
#include <convMultiBlock.h>
#include <util.h>

namespace gpu {
namespace op {

ConvMultiBlock::ConvMultiBlock(gpu::Device *dev, int inputChannel,
                               int inputHeight, int inputWidth,
                               int outputChannel, int maxConvSize,
                               bool instanceOn, int inputN, bool is_split)
    : gpu::op::Operation(dev, inputChannel, inputHeight, inputWidth),
      maxConvSize_(maxConvSize),
      instanceOn_(instanceOn),
      inputN_(inputN),
      is_split_(is_split) {
  int c, h, w, c2, h2, w2;

  conv3a_ = new gpu::op::ConvLayer(
      dev, inputChannel_, inputHeight_, inputWidth_, outputChannel, 3, 1, true,
      instanceOn, inputN_, ConvolutionMode::CONV_WINO);
  conv3a_->GetOutputSize(c, h, w);

  conv3b_ =
      new gpu::op::ConvLayer(dev, outputChannel, h, w, outputChannel, 3, 1,
                             true, instanceOn, 1, ConvolutionMode::CONV_WINO);
  conv3b_->GetOutputSize(c, h, w);

  if (maxConvSize_ >= 5) {
    conv5a_ = new gpu::op::ConvLayer(
        dev, inputChannel_, inputHeight_, inputWidth_, outputChannel, 5, 1,
        true, instanceOn, inputN_, ConvolutionMode::CONV_WINO, is_split_);
    conv5a_->GetOutputSize(c2, h2, w2);

    conv5b_ =
        new gpu::op::ConvLayer(dev, outputChannel, h2, w2, outputChannel, 5, 1,
                               true, instanceOn, 1, ConvolutionMode::CONV_WINO);
    conv5b_->GetOutputSize(c2, h2, w2);
    cat5_ = new gpu::op::Cat(dev, c, h, w, c2, h2, w2, 1);
    cat5_->GetOutputSize(c, h, w);
  }

  if (maxConvSize_ >= 7) {
    conv7a_ = new gpu::op::ConvLayer(
        dev, inputChannel_, inputHeight_, inputWidth_, outputChannel, 7, 1,
        true, instanceOn, inputN_, ConvolutionMode::CONV_IMPLICIT);
    conv7a_->GetOutputSize(c2, h2, w2);
    conv7b_ = new gpu::op::ConvLayer(dev, outputChannel, h2, w2, outputChannel,
                                     7, 1, true, instanceOn, 1,
                                     ConvolutionMode::CONV_IMPLICIT);
    conv7b_->GetOutputSize(c2, h2, w2);
    cat7_ = new gpu::op::Cat(dev, c, h, w, c2, h2, w2, 1);
    cat7_->GetOutputSize(c, h, w);
  }

  if (maxConvSize_ >= 9) {
    conv9a_ = new gpu::op::ConvLayer(
        dev, inputChannel_, inputHeight_, inputWidth_, outputChannel, 9, 1,
        true, instanceOn, inputN_, ConvolutionMode::CONV_IMPLICIT);
    conv9a_->GetOutputSize(c2, h2, w2);
    conv9b_ = new gpu::op::ConvLayer(dev, outputChannel, h2, w2, outputChannel,
                                     9, 1, true, instanceOn, 1,
                                     ConvolutionMode::CONV_IMPLICIT);
    conv9b_->GetOutputSize(c2, h2, w2);
    cat9_ = new gpu::op::Cat(dev, c, h, w, c2, h2, w2, 1);
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

void ConvMultiBlock::Forward(gpu::Memory *input, gpu::Memory *output,
                             gpu::Memory *l1, gpu::Memory *s1) {
  conv3a_->Forward(input, s1, l1);

  conv3b_->Forward(s1, output, l1);

  if (maxConvSize_ >= 5) {
    conv5a_->Forward(input, s1, l1);
    conv5b_->Forward(s1, s1, l1);

    cat5_->Forward(output, s1, output);
  }
  if (maxConvSize_ >= 7) {
    conv7a_->Forward(input, s1, l1);
    conv7b_->Forward(s1, s1, l1);
    cat7_->Forward(output, s1, output);
  }
  if (maxConvSize_ >= 9) {
    conv9a_->Forward(input, s1, l1);
    conv9b_->Forward(s1, s1, l1);
    cat9_->Forward(output, s1, output);
  }
}

}  // namespace op
}  // namespace gpu

namespace gpu {
namespace op {

ConvMultiBlock2::ConvMultiBlock2(gpu::Device *dev, int inputChannel,
                                 int inputHeight, int inputWidth,
                                 int outputChannel, int maxConvSize,
                                 bool instanceOn, int inputN, bool is_split)
    : gpu::op::Operation(dev, inputChannel, inputHeight, inputWidth),
      maxConvSize_(maxConvSize),
      instanceOn_(instanceOn),
      inputN_(inputN),
      is_split_(is_split) {
  int c, h, w, c2, h2, w2;

  conv3a_ = new gpu::op::ConvLayer2(
      dev, inputChannel_, inputHeight_, inputWidth_, outputChannel, 3, 1, true,
      instanceOn, inputN_, ConvolutionMode::CONV_WINO);
  conv3a_->GetOutputSize(c, h, w);

  conv3b_ =
      new gpu::op::ConvLayer(dev, outputChannel, h, w, outputChannel, 3, 1,
                             true, instanceOn, 1, ConvolutionMode::CONV_WINO);
  conv3b_->GetOutputSize(c, h, w);

  if (maxConvSize_ >= 5) {
    conv5a_ = new gpu::op::ConvLayer2(
        dev, inputChannel_, inputHeight_, inputWidth_, outputChannel, 5, 1,
        true, instanceOn, inputN_, ConvolutionMode::CONV_WINO);
    conv5a_->GetOutputSize(c2, h2, w2);

    conv5b_ =
        new gpu::op::ConvLayer(dev, outputChannel, h2, w2, outputChannel, 5, 1,
                               true, instanceOn, 1, ConvolutionMode::CONV_WINO);
    conv5b_->GetOutputSize(c2, h2, w2);
    cat5_ = new gpu::op::Cat(dev, c, h, w, c2, h2, w2, 1);
    cat5_->GetOutputSize(c, h, w);
  }

  if (maxConvSize_ >= 7) {
    conv7a_ = new gpu::op::ConvLayer2(
        dev, inputChannel_, inputHeight_, inputWidth_, outputChannel, 7, 1,
        true, instanceOn, inputN_, ConvolutionMode::CONV_IMPLICIT);
    conv7a_->GetOutputSize(c2, h2, w2);
    conv7b_ = new gpu::op::ConvLayer(dev, outputChannel, h2, w2, outputChannel,
                                     7, 1, true, instanceOn, 1,
                                     ConvolutionMode::CONV_IMPLICIT);
    conv7b_->GetOutputSize(c2, h2, w2);
    cat7_ = new gpu::op::Cat(dev, c, h, w, c2, h2, w2, 1);
    cat7_->GetOutputSize(c, h, w);
  }

  if (maxConvSize_ >= 9) {
    conv9a_ = new gpu::op::ConvLayer2(
        dev, inputChannel_, inputHeight_, inputWidth_, outputChannel, 9, 1,
        true, instanceOn, inputN_, ConvolutionMode::CONV_IMPLICIT);
    conv9a_->GetOutputSize(c2, h2, w2);
    conv9b_ = new gpu::op::ConvLayer(dev, outputChannel, h2, w2, outputChannel,
                                     9, 1, true, instanceOn, 1,
                                     ConvolutionMode::CONV_IMPLICIT);
    conv9b_->GetOutputSize(c2, h2, w2);
    cat9_ = new gpu::op::Cat(dev, c, h, w, c2, h2, w2, 1);
    cat9_->GetOutputSize(c, h, w);
  }

  outputChannel_ = c;
  outputHeight_ = h;
  outputWidth_ = w;
}

ConvMultiBlock2::~ConvMultiBlock2() {
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

void ConvMultiBlock2::LoadWeight(std::ifstream &fp) {
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

void ConvMultiBlock2::Forward(vector<gpu::Memory *> inputs,
                              vector<gpu::Memory *> outputs, gpu::Memory *l1,
                              gpu::Memory *s1) {
  // output
  conv3a_->Forward(inputs, outputs[0], l1, s1);

  conv3b_->Forward(outputs[0], outputs[0], l1);

  if (maxConvSize_ >= 5) {
    conv5a_->Forward(inputs, outputs[1], l1, s1);
    conv5b_->Forward(outputs[1], outputs[1], l1);
  }
  if (maxConvSize_ >= 7) {
    conv7a_->Forward(inputs, outputs[2], l1, s1);
    conv7b_->Forward(outputs[2], outputs[2], l1);
  }
  if (maxConvSize_ >= 9) {
    conv9a_->Forward(inputs, outputs[3], l1, s1);
    conv9b_->Forward(outputs[3], outputs[3], l1);
  }
}

}  // namespace op
}  // namespace gpu
