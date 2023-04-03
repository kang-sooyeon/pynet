
#include <cat.h>
#include <convLayer.h>
#include <util.h>

namespace gpu {
namespace op {

ConvLayer::ConvLayer(gpu::Device *dev, int inputChannel, int inputHeight,
                     int inputWidth, int outputChannel, int kernelSize,
                     int stride, bool reluOn, bool instanceOn, int inputN,
                     ConvolutionMode mode, bool is_split)
    : gpu::op::Operation(dev, inputChannel, inputHeight, inputWidth),
      kernelSize_(kernelSize),
      mode_(mode),
      stride_(stride),
      reluOn_(reluOn),
      instanceOn_(instanceOn),
      inputN_(inputN),
      is_split_(is_split) {
  int c, h, w;

  // reflectionpad
  reflectionPadding_ = kernelSize_ / 2;
  reflectionPad_ = new ReflectionPad2d(dev, inputChannel, inputHeight,
                                       inputWidth, reflectionPadding_);
  reflectionPad_->GetOutputSize(c, h, w);

  // convolution2d
  conv2d_ = new Convolution2d(dev, c, h, w, outputChannel, kernelSize_, stride_,
                              mode_, is_split_);

  conv2d_->GetOutputSize(c, h, w);

  // instanceNorm2d
  if (instanceOn_) {
    instance_ = new InstanceNorm2d(dev, c, h, w, true, reluOn_);
    instance_->GetOutputSize(c, h, w);
  }

  // leakyReLU
  if (reluOn_) {
    relu_ = new LeakyReLU(dev, c, h, w);
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

void ConvLayer::Forward(gpu::Memory *input, gpu::Memory *output,
                        gpu::Memory *l1) {
  reflectionPad_->Forward(input, l1);

  conv2d_->Forward(l1, output);

  if (instanceOn_) {
    instance_->Forward(output, l1);
  } else {
    if (reluOn_) {
      relu_->Forward(output, output);
    }
  }
}
}  // namespace op
}  // namespace gpu

namespace gpu {
namespace op {

ConvLayer2::ConvLayer2(gpu::Device *dev, int inputChannel, int inputHeight,
                       int inputWidth, int outputChannel, int kernelSize,
                       int stride, bool reluOn, bool instanceOn, int inputN,
                       ConvolutionMode mode, bool is_split)
    : gpu::op::Operation(dev, inputChannel, inputHeight, inputWidth),
      kernelSize_(kernelSize),
      stride_(stride),
      reluOn_(reluOn),
      instanceOn_(instanceOn),
      inputN_(inputN),
      mode_(mode),
      is_split_(is_split) {
  int c, h, w;

  // reflectionpad
  reflectionPadding_ = kernelSize_ / 2;
  reflectionPad_ = new ReflectionPad2d(dev, inputChannel / inputN_, inputHeight,
                                       inputWidth, reflectionPadding_);
  reflectionPad_->GetOutputSize(c, h, w);

  // convolution2d
  for (int i = 0; i < inputN_; i++) {
    conv2d_.push_back(new Convolution2d2(dev, c, h, w, outputChannel,
                                         kernelSize_, stride_, mode_));
  }
  conv2d_[0]->GetOutputSize(c, h, w);

  // convolution bias
  biasSize_ = outputChannel * sizeof(DATA_TYPE);
  bias_ = new gpu::Memory(dev_, biasSize_);

  // instanceNorm2d
  if (instanceOn_) {
    instance_ = new InstanceNorm2d(dev, c, h, w, true, reluOn_);
    instance_->GetOutputSize(c, h, w);
  }

  // leakyReLU
  if (reluOn_) {
    relu_ = new LeakyReLU(dev, c, h, w);
    relu_->GetOutputSize(c, h, w);
  }

  outputChannel_ = c;
  outputHeight_ = h;
  outputWidth_ = w;

  cl_int err = 0;
  kernel_ = clCreateKernel(dev_->ClDnnHandle()->program, "addBias", &err);
  CheckCl(err);
}

void ConvLayer2::LoadWeight(std::ifstream &fp) {
  for (int i = 0; i < inputN_; i++) {
    conv2d_[i]->LoadWeight(fp);
  }

  char *bias_H = (char *)malloc(biasSize_);
  fp.read(bias_H, biasSize_);
  bias_->Set(bias_H);
  clFinish(dev_->ClDnnHandle()->queue);
  delete bias_H;

  if (instanceOn_) {
    instance_->LoadWeight(fp);
  }
}

ConvLayer2::~ConvLayer2() {
  for (int i = 0; i < inputN_; i++) delete conv2d_[i];

  delete bias_;

  if (instanceOn_) delete instance_;
}

void ConvLayer2::Forward(vector<gpu::Memory *> inputs, gpu::Memory *output,
                         gpu::Memory *l1, gpu::Memory *s1) {
  reflectionPad_->Forward(inputs[0], l1);
  conv2d_[0]->Forward(l1, output);

  for (int i = 1; i < inputN_; i++) {
    reflectionPad_->Forward(inputs[i], l1);
    conv2d_[i]->Forward(l1, s1);

    // element-wise sum
    Add(s1, output);
  }
  AddBias(bias_, output);

  if (instanceOn_) {
    instance_->Forward(output, l1);
  } else {
    if (reluOn_) {
      relu_->Forward(output, output);
    }
  }
}

void ConvLayer2::AddBias(gpu::Memory *bias, gpu::Memory *input) {
  cl_mem biasMem = bias->Ptr();
  cl_mem inputMem = input->Ptr();

  cl_int err = 0;
  err = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &inputMem);
  err = clSetKernelArg(kernel_, 1, sizeof(cl_mem), &biasMem);
  err = clSetKernelArg(kernel_, 2, sizeof(cl_int), &outputChannel_);
  err = clSetKernelArg(kernel_, 3, sizeof(cl_int), &outputHeight_);
  err = clSetKernelArg(kernel_, 4, sizeof(cl_int), &outputWidth_);
  CheckCl(err);

  size_t globalSize[3] = {(size_t)outputWidth_, (size_t)outputHeight_,
                          (size_t)outputChannel_};
  size_t localSize[3] = {16, 4, 4};

  globalSize[0] =
      (globalSize[0] + localSize[0] - 1) / localSize[0] * localSize[0];
  globalSize[1] =
      (globalSize[1] + localSize[1] - 1) / localSize[1] * localSize[1];
  globalSize[2] =
      (globalSize[2] + localSize[2] - 1) / localSize[2] * localSize[2];

  CheckCl(clEnqueueNDRangeKernel(dev_->ClDnnHandle()->queue, kernel_, 3, NULL,
                                 globalSize, localSize, 0, NULL, NULL));
  input->SetUsedSize(outputChannel_ * outputHeight_ * outputWidth_ *
                     sizeof(DATA_TYPE));
}

}  // namespace op
}  // namespace gpu
