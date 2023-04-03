#include <pynet.h>
#include <util.h>

static cl_kernel addKernel_;

void Add(gpu::Memory *input, gpu::Memory *output) {
  cl_mem inMem = input->Ptr();
  cl_mem outMem = output->Ptr();

  int num1 = input->Size() / sizeof(DATA_TYPE);
  int num2 = output->Size() / sizeof(DATA_TYPE);

  if (num1 != num2) {
    cout << "[Add] : invalid size" << endl;
    exit(-1);
  }

  cl_int err = 0;
  err = clSetKernelArg(addKernel_, 0, sizeof(cl_mem), &inMem);
  err = clSetKernelArg(addKernel_, 1, sizeof(cl_mem), &outMem);
  err = clSetKernelArg(addKernel_, 2, sizeof(cl_int), &num1);
  CheckCl(err);

  size_t globalSize[] = {(size_t)num1};
  size_t localSize[] = {(size_t)256};

  globalSize[0] =
      (globalSize[0] + localSize[0] - 1) / localSize[0] * localSize[0];
  CheckCl(clEnqueueNDRangeKernel(input->Dev()->ClDnnHandle()->queue, addKernel_,
                                 1, NULL, globalSize, localSize, 0, NULL,
                                 NULL));
}

void Add2(vector<gpu::Memory *> inputs, vector<gpu::Memory *> outputs) {
  for (int i = 0; i < 4; i++) {
    Add(inputs[i], outputs[i]);
  }
}

namespace gpu {
namespace op {

PyNET::PyNET(gpu::Device *dev, int inputChannel, int inputHeight,
             int inputWidth, int level, bool instanceOn, bool instanceOnL1)
    : gpu::op::Operation(dev, inputChannel, inputHeight, inputWidth),
      level_(level),
      instanceOn_(instanceOn),
      instanceOnL1_(instanceOnL1) {
  int c, h, w, c2, h2, w2;

  // D1
  convL1D1_ = new gpu::op::ConvMultiBlock(dev, 4, inputHeight_, inputWidth_, 32,
                                          3, false, 1);
  convL1D1_->GetOutputSize(c, h, w);
  convL1D1_Size = c * h * w * sizeof(DATA_TYPE);

  pool1_ = new gpu::op::MaxPool2d(dev, c, h, w, 2, 2);
  pool1_->GetOutputSize(c, h, w);

  convL2D1_ = new gpu::op::ConvMultiBlock(dev, 32, h, w, 64, 3, instanceOn_, 1);
  convL2D1_->GetOutputSize(c, h, w);
  convL2D1_Size = c * h * w * sizeof(DATA_TYPE);
  pool2_ = new gpu::op::MaxPool2d(dev, c, h, w, 2, 2);
  pool2_->GetOutputSize(c, h, w);

  convL3D1_ = new gpu::op::ConvMultiBlock(dev, 64, h, w, 128, 3, instanceOn, 1);
  convL3D1_->GetOutputSize(c, h, w);
  convL3D1_Size = c * h * w * sizeof(DATA_TYPE);
  pool3_ = new gpu::op::MaxPool2d(dev, c, h, w, 2, 2);
  pool3_->GetOutputSize(c, h, w);

  convL4D1_ =
      new gpu::op::ConvMultiBlock(dev, 128, h, w, 256, 3, instanceOn, 1);
  convL4D1_->GetOutputSize(c, h, w);
  convL4D1_Size = c * h * w * sizeof(DATA_TYPE);
  pool4_ = new gpu::op::MaxPool2d(dev, c, h, w, 2, 2);
  pool4_->GetOutputSize(c, h, w);

  // level 5
  convL5D1_ =
      new gpu::op::ConvMultiBlock(dev, 256, h, w, 512, 3, instanceOn, 1);
  convL5D1_->GetOutputSize(c, h, w);
  convL5D2_ =
      new gpu::op::ConvMultiBlock(dev, 512, h, w, 512, 3, instanceOn, 1);
  convL5D2_->GetOutputSize(c, h, w);
  convL5D3_ =
      new gpu::op::ConvMultiBlock(dev, 512, h, w, 512, 3, instanceOn, 1);
  convL5D3_->GetOutputSize(c, h, w);
  convL5D4_ =
      new gpu::op::ConvMultiBlock(dev, 512, h, w, 512, 3, instanceOn, 1);
  convL5D4_->GetOutputSize(c, h, w);
  convT4a_ = new gpu::op::UpsampleConvLayer(dev, 512, h, w, 256, 3,
                                            ConvolutionMode::CONV_WINO);
  convT4b_ = new gpu::op::UpsampleConvLayer(dev, 512, h, w, 256, 3,
                                            ConvolutionMode::CONV_WINO);
  convL5Out_ = new gpu::op::ConvLayer(dev, 512, h, w, 3, 3, 1, false, false, 1,
                                      ConvolutionMode::CONV_WINO);
  convL5Out_->GetOutputSize(c, h, w);
  outputL5_ = new gpu::op::Sigmoid(dev, c, h, w);

  // level 4
  convL4D1_->GetOutputSize(c, h, w);
  convT4a_->GetOutputSize(c2, h2, w2);
  convL4D2_ = new gpu::op::Cat(dev, c, h, w, c2, h2, w2, 1);
  convL4D2_->GetOutputSize(c, h, w);
  convL4D3_ =
      new gpu::op::ConvMultiBlock(dev, 512, h, w, 256, 3, instanceOn, 1);
  convL4D3_->GetOutputSize(c, h, w);
  convL4D4_ =
      new gpu::op::ConvMultiBlock(dev, 256, h, w, 256, 3, instanceOn, 1);
  convL4D4_->GetOutputSize(c, h, w);
  convL4D5_ =
      new gpu::op::ConvMultiBlock(dev, 256, h, w, 256, 3, instanceOn, 1);
  convL4D5_->GetOutputSize(c, h, w);
  convL4D6_ =
      new gpu::op::ConvMultiBlock(dev, 256, h, w, 256, 3, instanceOn, 1);
  convL4D6_->GetOutputSize(c, h, w);
  convT4b_->GetOutputSize(c2, h2, w2);
  convL4D7_ = new gpu::op::Cat(dev, c, h, w, c2, h2, w2, 1);
  convL4D7_->GetOutputSize(c, h, w);
  convL4D8_ =
      new gpu::op::ConvMultiBlock(dev, 512, h, w, 256, 3, instanceOn, 1);
  convL4D8_->GetOutputSize(c, h, w);
  convT3a_ = new gpu::op::UpsampleConvLayer(dev, 256, h, w, 128, 3,
                                            ConvolutionMode::CONV_WINO);
  convT3b_ = new gpu::op::UpsampleConvLayer(dev, 256, h, w, 128, 3,
                                            ConvolutionMode::CONV_WINO);
  convL4Out_ = new gpu::op::ConvLayer(dev, 256, h, w, 3, 3, 1, false, false, 1,
                                      ConvolutionMode::CONV_WINO);
  convL4Out_->GetOutputSize(c, h, w);
  outputL4_ = new gpu::op::Sigmoid(dev, c, h, w);

  // level 3
  convL3D1_->GetOutputSize(c, h, w);
  convT3a_->GetOutputSize(c2, h2, w2);
  convL3D2_ = new gpu::op::Cat(dev, c, h, w, c2, h2, w2, 1);
  convL3D2_->GetOutputSize(c, h, w);
  convL3D3_ =
      new gpu::op::ConvMultiBlock(dev, 256, h, w, 128, 5, instanceOn, 1);
  convL3D3_->GetOutputSize(c, h, w);
  convL3D4_ =
      new gpu::op::ConvMultiBlock(dev, 256, h, w, 128, 5, instanceOn, 1);
  convL3D4_->GetOutputSize(c, h, w);
  convL3D5_ =
      new gpu::op::ConvMultiBlock(dev, 256, h, w, 128, 5, instanceOn, 1);
  convL3D5_->GetOutputSize(c, h, w);
  convL3D6_ =
      new gpu::op::ConvMultiBlock(dev, 256, h, w, 128, 5, instanceOn, 1);

  convL3D6_->GetOutputSize(c, h, w);
  convL3D1_->GetOutputSize(c2, h2, w2);
  convL3D7_1_ = new gpu::op::Cat(dev, c, h, w, c2, h2, w2, 1);
  convL3D7_1_->GetOutputSize(c, h, w);
  convT3b_->GetOutputSize(c2, h2, w2);
  convL3D7_ = new gpu::op::Cat(dev, c, h, w, c2, h2, w2, 1);

  convL3D7_->GetOutputSize(c, h, w);
  convL3D8_ =
      new gpu::op::ConvMultiBlock(dev, 512, h, w, 128, 3, instanceOn, 1);
  convL3D8_->GetOutputSize(c, h, w);
  convT2a_ = new gpu::op::UpsampleConvLayer(dev, 128, h, w, 64, 3,
                                            ConvolutionMode::CONV_WINO);
  convT2b_ = new gpu::op::UpsampleConvLayer(dev, 128, h, w, 64, 3,
                                            ConvolutionMode::CONV_WINO);
  convL3Out_ = new gpu::op::ConvLayer(dev, 128, h, w, 3, 3, 1, false, false, 1,
                                      ConvolutionMode::CONV_WINO);
  convL3Out_->GetOutputSize(c, h, w);
  outputL3_ = new gpu::op::Sigmoid(dev, c, h, w);

  // level 2
  convL2D1_->GetOutputSize(c, h, w);
  convT2a_->GetOutputSize(c2, h2, w2);
  convL2D2_ = new gpu::op::Cat(dev, c, h, w, c2, h2, w2, 1);
  convL2D2_->GetOutputSize(c, h, w);
  convL2D3_ = new gpu::op::ConvMultiBlock(dev, 128, h, w, 64, 5, instanceOn, 1);
  convL2D3_->GetOutputSize(c, h, w);
  convL2D1_->GetOutputSize(c2, h2, w2);
  convL2D4_ = new gpu::op::Cat(dev, c, h, w, c2, h2, w2, 1);
  convL2D4_->GetOutputSize(c, h, w);
  convL2D5_ = new gpu::op::ConvMultiBlock(dev, 192, h, w, 64, 7, instanceOn, 1,
                                          true);  // wino split

  convL2D5_->GetOutputSize(c, h, w);
  convL2D6_ = new gpu::op::ConvMultiBlock(dev, 192, h, w, 64, 7, instanceOn, 1,
                                          true);  // wino split
  convL2D6_->GetOutputSize(c, h, w);
  convL2D7_ = new gpu::op::ConvMultiBlock(dev, 192, h, w, 64, 7, instanceOn, 1,
                                          true);  // wino split

  convL2D7_->GetOutputSize(c, h, w);
  convL2D8_ = new gpu::op::ConvMultiBlock(dev, 192, h, w, 64, 7, instanceOn, 1,
                                          true);  // wino split

  convL2D8_->GetOutputSize(c, h, w);
  convL2D1_->GetOutputSize(c2, h2, w2);
  convL2D9_ = new gpu::op::Cat(dev, c, h, w, c2, h2, w2, 1);
  convL2D9_->GetOutputSize(c, h, w);
  convL2D10_ = new gpu::op::ConvMultiBlock(dev, 256, h, w, 64, 5, instanceOn, 1,
                                           true);  // wino split

  convL2D10_->GetOutputSize(c, h, w);
  convT2b_->GetOutputSize(c2, h2, w2);
  convL2D11_ = new gpu::op::Cat(dev, c, h, w, c2, h2, w2, 1);
  convL2D11_->GetOutputSize(c, h, w);
  convL2D12_ =
      new gpu::op::ConvMultiBlock(dev, 192, h, w, 64, 3, instanceOn, 1);
  convL2D12_->GetOutputSize(c, h, w);
  convT1a_ = new gpu::op::UpsampleConvLayer(dev, 64, h, w, 32, 3,
                                            ConvolutionMode::CONV_WINO);
  convT1b_ = new gpu::op::UpsampleConvLayer(dev, 64, h, w, 32, 3,
                                            ConvolutionMode::CONV_WINO);
  convL2Out_ = new gpu::op::ConvLayer(dev, 64, h, w, 3, 3, 1, false, false, 1,
                                      ConvolutionMode::CONV_WINO);
  convL2Out_->GetOutputSize(c, h, w);
  outputL2_ = new gpu::op::Sigmoid(dev, c, h, w);

  // level 1
  convL1D1_->GetOutputSize(c, h, w);
  convT1a_->GetOutputSize(c2, h2, w2);

  convL1D2_ = new gpu::op::Cat(dev, c, h, w, c2, h2, w2, 1);
  convL1D2_->GetOutputSize(c, h, w);

  convL1D3_ = new gpu::op::ConvMultiBlock2(dev, 64, h, w, 32, 5, false, 2);
  convL1D3_->GetOutputSize(c, h, w);
  convL1D1_->GetOutputSize(c2, h2, w2);
  convL1D4_ = new gpu::op::Cat(dev, c, h, w, c2, h2, w2, 1);
  convL1D4_->GetOutputSize(c, h, w);

  convL1D5_ =
      new gpu::op::ConvMultiBlock2(dev, 96, h, w, 32, 7, instanceOnL1, 3);
  convL1D5_->GetOutputSize(c, h, w);

  convL1D6_ =
      new gpu::op::ConvMultiBlock2(dev, 96, h, w, 32, 9, instanceOnL1, 3);
  convL1D6_->GetOutputSize(c, h, w);
  convL1D7_ =
      new gpu::op::ConvMultiBlock2(dev, 128, h, w, 32, 9, instanceOnL1, 4);
  convL1D7_->GetOutputSize(c, h, w);
  convL1D8_ =
      new gpu::op::ConvMultiBlock2(dev, 128, h, w, 32, 9, instanceOnL1, 4);
  convL1D8_->GetOutputSize(c, h, w);
  convL1D9_ =
      new gpu::op::ConvMultiBlock2(dev, 128, h, w, 32, 9, instanceOnL1, 4);
  convL1D9_->GetOutputSize(c, h, w);

  convL1D10_ =
      new gpu::op::ConvMultiBlock2(dev, 128, h, w, 32, 7, instanceOnL1, 4);
  convL1D10_->GetOutputSize(c, h, w);
  convL1D1_->GetOutputSize(c2, h2, w2);
  convL1D11_ = new gpu::op::Cat(dev, c, h, w, c2, h2, w2, 1);
  convL1D11_->GetOutputSize(c, h, w);
  convL1D12_ =
      new gpu::op::ConvMultiBlock2(dev, 128, h, w, 32, 5, instanceOnL1, 4);

  convL1D12_->GetOutputSize(c, h, w);
  convT1b_->GetOutputSize(c2, h2, w2);
  convL1D13_1_ = new gpu::op::Cat(dev, c, h, w, c2, h2, w2, 1);
  convL1D13_1_->GetOutputSize(c, h, w);
  convL1D1_->GetOutputSize(c2, h2, w2);
  convL1D13_ = new gpu::op::Cat(dev, c, h, w, c2, h2, w2, 1);

  convL1D13_->GetOutputSize(c, h, w);
  convL1D14_ = new gpu::op::ConvMultiBlock2(dev, 128, h, w, 32, 3, false, 4);
  convL1D14_->GetOutputSize(c, h, w);

  convT0_ = new gpu::op::UpsampleConvLayer2(dev, 32, h, w, 16, 3, 2,
                                            ConvolutionMode::CONV_WINO);

  convL1Out_ = new gpu::op::ConvLayer(dev, 32, h, w, 3, 3, 1, false, false, 1,
                                      ConvolutionMode::CONV_WINO);
  convL1Out_->GetOutputSize(c, h, w);
  outputL1_ = new gpu::op::Sigmoid(dev, c, h, w);

  // level 0
  convT0_->GetOutputSize(c, h, w);
  convL0D1_ = new gpu::op::ConvLayer(dev, 16, h, w, 3, 3, 1, false, false, 1,
                                     ConvolutionMode::CONV_WINO);
  convL0D1_->GetOutputSize(c, h, w);
  outputL0_ = new gpu::op::Sigmoid(dev, c, h, w);

  // pynet outputsize
  if (level == 5)
    outputL5_->GetOutputSize(outputChannel_, outputHeight_, outputWidth_);
  else if (level == 4)
    outputL4_->GetOutputSize(outputChannel_, outputHeight_, outputWidth_);
  else if (level == 3)
    outputL3_->GetOutputSize(outputChannel_, outputHeight_, outputWidth_);
  else if (level == 2)
    outputL2_->GetOutputSize(outputChannel_, outputHeight_, outputWidth_);
  else if (level == 1)
    outputL1_->GetOutputSize(outputChannel_, outputHeight_, outputWidth_);
  else if (level == 0)
    outputL0_->GetOutputSize(outputChannel_, outputHeight_, outputWidth_);

  // add kernel
  cl_int err = 0;
  addKernel_ = clCreateKernel(dev_->ClDnnHandle()->program, "add", &err);
  splitKernel_ = clCreateKernel(dev_->ClDnnHandle()->program, "split", &err);
  CheckCl(err);

  convL2D1 = new gpu::Memory(dev_);
  convL3D1 = new gpu::Memory(dev_);
  convL4D1 = new gpu::Memory(dev_);
  convTb = new gpu::Memory(dev_);

  for (int i = 0; i < 4; i++) {
    inputs.push_back(new gpu::Memory(dev_));
    outputs.push_back(new gpu::Memory(dev_));
  }

  convL1D1_H = (DATA_TYPE *)malloc(convL1D1_Size);
  convTb_H = (DATA_TYPE *)malloc(smallSize);
}

void PyNET::LoadWeight(char *weightFile) {
  std::ifstream fp(weightFile, std::ios::binary);
  if (!fp) {
    cout << "cannot find weignt file" << endl;
    exit(-1);
  }

  convL1D1_->LoadWeight(fp);
  convL2D1_->LoadWeight(fp);
  convL3D1_->LoadWeight(fp);
  convL4D1_->LoadWeight(fp);

  // level 5
  convL5D1_->LoadWeight(fp);
  convL5D2_->LoadWeight(fp);
  convL5D3_->LoadWeight(fp);
  convL5D4_->LoadWeight(fp);
  convT4a_->LoadWeight(fp);
  convT4b_->LoadWeight(fp);
  convL5Out_->LoadWeight(fp);

  // level 4
  convL4D3_->LoadWeight(fp);
  convL4D4_->LoadWeight(fp);
  convL4D5_->LoadWeight(fp);
  convL4D6_->LoadWeight(fp);
  convL4D8_->LoadWeight(fp);
  convT3a_->LoadWeight(fp);
  convT3b_->LoadWeight(fp);
  convL4Out_->LoadWeight(fp);

  // level 3
  convL3D3_->LoadWeight(fp);
  convL3D4_->LoadWeight(fp);
  convL3D5_->LoadWeight(fp);
  convL3D6_->LoadWeight(fp);
  convL3D8_->LoadWeight(fp);
  convT2a_->LoadWeight(fp);
  convT2b_->LoadWeight(fp);
  convL3Out_->LoadWeight(fp);

  // level 2
  convL2D3_->LoadWeight(fp);
  convL2D5_->LoadWeight(fp);
  convL2D6_->LoadWeight(fp);
  convL2D7_->LoadWeight(fp);
  convL2D8_->LoadWeight(fp);
  convL2D10_->LoadWeight(fp);
  convL2D12_->LoadWeight(fp);
  convT1a_->LoadWeight(fp);
  convT1b_->LoadWeight(fp);
  convL2Out_->LoadWeight(fp);

  // level 1
  convL1D3_->LoadWeight(fp);
  convL1D5_->LoadWeight(fp);
  convL1D6_->LoadWeight(fp);
  convL1D7_->LoadWeight(fp);
  convL1D8_->LoadWeight(fp);
  convL1D9_->LoadWeight(fp);
  convL1D10_->LoadWeight(fp);
  convL1D12_->LoadWeight(fp);
  convL1D14_->LoadWeight(fp);
  convL1Out_->LoadWeight(fp);
  convT0_->LoadWeight(fp);

  // level 0
  convL0D1_->LoadWeight(fp);

  fp.close();
}

void PyNET::Forward(gpu::Memory *_l1, gpu::Memory *_l2, gpu::Memory *_l3,
                    gpu::Memory *_s1, gpu::Memory **_s2) {
  convL2D1->CreateBuffer(convL2D1_Size);
  convL3D1->CreateBuffer(convL3D1_Size);
  convL4D1->CreateBuffer(convL4D1_Size);
  convTb->CreateBuffer(smallSize);

  clFinish(dev_->ClDnnHandle()->queue);

  l1 = _l1;
  l2 = _l2;
  l3 = _l3;
  s1 = _s1;
  s2 = *_s2;

  convL1D1_->Forward(l1, l3, l2, s1);

  //////////// convL1D1 swap out
  cl_int status;
  clFinish(dev_->ClDnnHandle()->queue);
  float *tmp = (float *)clEnqueueMapBuffer(
      dev_->ClDnnHandle()->queue, l3->Ptr(), CL_TRUE, CL_MAP_WRITE, 0,
      convL1D1_Size, 0, NULL, NULL, &status);
  clFinish(dev_->ClDnnHandle()->queue);
  memcpy(convL1D1_H, tmp, convL1D1_Size);

  pool1_->Forward(l3, l1);
  convL2D1_->Forward(l1, convL2D1, l2, s2);
  pool2_->Forward(convL2D1, l1);
  convL3D1_->Forward(l1, convL3D1, l2, s1);
  pool3_->Forward(convL3D1, l1);
  convL4D1_->Forward(l1, convL4D1, l2, s1);
  pool4_->Forward(convL4D1, l1);

  level5();

  if (level_ < 5) {
    level4();
  }
  if (level_ < 4) {
    level3();
  }
  if (level_ < 3) {
    level2();
  }
  if (level_ < 2) {
    level1();
  }
  if (level_ < 1) {
    level0();
  }

  clFinish(dev_->ClDnnHandle()->queue);
  if (level_ >= 3) {
    convL4D1->Release();
    convL3D1->Release();
    convL2D1->Release();
    convTb->Release();
    s1->Release();
  } else if (level_ == 2) {
    s1->Release();
    inputs[0]->Release();
    inputs[1]->Release();
  } else {
    outputs[0]->Release();
    outputs[1]->Release();
  }
  l1->Release();
  l2->Release();
  l3->Release();

  // have to open
  *_s2 = s2;
}

void PyNET::level5() {
  convL5D1_->Forward(l1, l3, l2, s1);
  convL5D2_->Forward(l3, l1, l2, s1);
  convL5D3_->Forward(l1, l3, l2, s1);
  convL5D4_->Forward(l3, s1, l1, l2);
  convT4b_->Forward(s1, convTb, l1, l2);
  convT4a_->Forward(s1, l3, l1, l2);
  convL5Out_->Forward(s1, s2, l1);

  // Unused outputs, follow original pytorch implements
  outputL5_->Forward(s2, s2);
}

void PyNET::level4() {
  convL4D2_->Forward(convL4D1, l3, l1);
  convL4D3_->Forward(l1, l3, l2, s1);
  convL4D4_->Forward(l3, l1, l2, s1);
  Add(l3, l1);

  convL4D5_->Forward(l1, l3, l2, s1);
  Add(l1, l3);

  convL4D6_->Forward(l3, l1, l2, s1);
  convL4D7_->Forward(l1, convTb, l1);
  convL4D8_->Forward(l1, s1, l2, l3);
  convT3b_->Forward(s1, convTb, l1, l2);
  convT3a_->Forward(s1, l3, l1, l2);
  convL4Out_->Forward(s1, s2, l1);

  // Unused outputs, follow original pytorch implements
  outputL4_->Forward(s2, s2);
}

void PyNET::level3() {
  convL3D2_->Forward(convL3D1, l3, l1);
  convL3D3_->Forward(l1, l3, l2, s1);
  convL3D4_->Forward(l1, l3, l2, s1);
  Add(l1, l3);

  convL3D5_->Forward(l3, l1, l2, s1);
  Add(l3, l1);

  convL3D6_->Forward(l1, l3, l2, s1);
  convL3D7_1_->Forward(l3, convL3D1, l3);
  convL3D7_->Forward(l3, convTb, l3);
  convL3D8_->Forward(l3, s1, l1, l2);
  convT2b_->Forward(s1, convTb, l1, l2);
  convT2a_->Forward(s1, l3, l1, l2);
  convL3Out_->Forward(s1, s2, l1);

  // Unused outputs, follow original pytorch implements
  outputL3_->Forward(s2, s2);
}

void PyNET::level2() {
  clFinish(dev_->ClDnnHandle()->queue);
  convL4D1->Release();
  convL3D1->Release();
  clFinish(dev_->ClDnnHandle()->queue);

  convL2D2_->Forward(convL2D1, l3, l1);
  convL2D3_->Forward(l1, l3, l2, s1);
  convL2D4_->Forward(l3, convL2D1, l3);
  convL2D5_->Forward(l3, l1, l2, s1);
  Add(l3, l1);

  convL2D6_->Forward(l1, l3, l2, s1);
  Add(l1, l3);

  convL2D7_->Forward(l3, l1, l2, s1);
  Add(l3, l1);

  convL2D8_->Forward(l1, l3, l2, s1);
  convL2D9_->Forward(l3, convL2D1, l3);
  convL2D10_->Forward(l3, l1, l2, s1);
  convL2D11_->Forward(l1, convTb, l1);
  convL2D12_->Forward(l1, s1, l2, l3);
  convT1b_->Forward(s1, convTb, l1, l2);

  ///// convT1b swap out
  cl_int status;
  convT_Size = convTb->Size();
  float *tmp = (float *)clEnqueueMapBuffer(
      dev_->ClDnnHandle()->queue, convTb->Ptr(), CL_TRUE, CL_MAP_WRITE, 0,
      convT_Size, 0, NULL, NULL, &status);
  clFinish(dev_->ClDnnHandle()->queue);
  memcpy(convTb_H, tmp, convT_Size);

  ///////////////////// split small buffer
  clFinish(dev_->ClDnnHandle()->queue);
  convL2D1->Release();
  convTb->Release();

  inputs[0]->CreateBuffer(smallSize);
  inputs[1]->CreateBuffer(smallSize);

  clFinish(dev_->ClDnnHandle()->queue);
  /////////////////////// convL1D1 swap in
  inputs[0]->SetUsedSize(convL1D1_Size);
  inputs[0]->Set(convL1D1_H);

  clFinish(dev_->ClDnnHandle()->queue);

  convT1a_->Forward(s1, inputs[1], l1, l2);
  convL2Out_->Forward(s1, s2, l1);

  // Unused outputs, follow original pytorch implements
  outputL2_->Forward(s2, s2);
}

void PyNET::level1() {
  clFinish(dev_->ClDnnHandle()->queue);

  l1->Release();
  l2->Release();
  l3->Release();

  inputs[2]->CreateBuffer(smallSize);
  inputs[3]->CreateBuffer(smallSize);
  outputs[0]->CreateBuffer(smallSize);
  outputs[1]->CreateBuffer(smallSize);
  outputs[2]->CreateBuffer(smallSize);
  outputs[3]->CreateBuffer(smallSize);
  clFinish(dev_->ClDnnHandle()->queue);
  ///////////////////

  ////////////////////////////////////
  // inputs[0]~[3]  : 388 M
  // outputs[0]~[3] : 388 M
  // s1, s2 	  : 388 M
  /////////////////////////////////////

  ///////////////////////////////////// convL1D1  swap in
  // printf("*********** start Level1 **************\n");
  clFinish(dev_->ClDnnHandle()->queue);
  outputs[2]->SetUsedSize(convL1D1_Size);
  outputs[2]->Set(convL1D1_H);
  clFinish(dev_->ClDnnHandle()->queue);

  convL1D3_->Forward(inputs, outputs,  // inputs[0~1] --> outputs[0~1]
                     s1, s2);

  convL1D5_->Forward(outputs, inputs,  //  outputs[0~2] --> inputs[0~2]
                     s1, s2);

  convL1D6_->Forward(inputs, outputs,  // inputs[0~2] --> outputs[0~3]
                     s1, s2);

  convL1D7_->Forward(outputs, inputs,  // outputs[0~3] --> inputs[0~3]
                     s1, s2);
  Add2(outputs, inputs);

  convL1D8_->Forward(inputs, outputs,  // inputs[0~3] --> outputs[0~3]
                     s1, s2);
  Add2(inputs, outputs);

  convL1D9_->Forward(outputs, inputs,  // outputs[0~3] --> inputs[0~3]
                     s1, s2);
  Add2(outputs, inputs);

  //////////////////////////////// convL1D1  swap in --> convL1D12_ input
  clFinish(dev_->ClDnnHandle()->queue);
  outputs[3]->SetUsedSize(convL1D1_Size);
  outputs[3]->Set(convL1D1_H);

  clFinish(dev_->ClDnnHandle()->queue);
  convL1D10_->Forward(inputs, outputs,  // inputs[0~3] --> outputs[0~3]
                      s1, s2);

  ///////////////////////convT1b, convL1D1 swap in --> convL1D14_ input
  clFinish(dev_->ClDnnHandle()->queue);
  inputs[2]->SetUsedSize(convT_Size);
  inputs[2]->Set(convTb_H);

  inputs[3]->SetUsedSize(convL1D1_Size);
  inputs[3]->Set(convL1D1_H);
  clFinish(dev_->ClDnnHandle()->queue);
  ////////////////////////////////

  convL1D12_->Forward(outputs, inputs,  // outputs[0~3] --> inputs[0~1]
                      s1, s2);

  convL1D14_->Forward(inputs, outputs,  // inputs[0~3] --> outputs[0]
                      s1, s2);

  convL1Out_->Forward(outputs[0], s2, s1);
  outputL1_->Forward(s2, s2);

  Split(outputs);  // outputs[0] --> outputs[0], outputs[1]
  clFinish(dev_->ClDnnHandle()->queue);

  outputs[2]->Release();
  outputs[3]->Release();
  inputs[0]->Release();
  inputs[1]->Release();
  inputs[2]->Release();
  inputs[3]->Release();
  s1->Release();

  l1->CreateBuffer(largeSize);
  l2->CreateBuffer(largeSize);
  l3->CreateBuffer(largeSize);
  ////////////////////////////////////
  // outputs[0]~[1] : 388 M
  // s2             : 388 M
  // l1, l2, l3     : 788 M
  /////////////////////////////////////

  convT0_->Forward(outputs, l1, l2, l3);  // **
}

// Hidden level, to match(resize) final outputs to original input image size. 
void PyNET::level0() {
  convL0D1_->Forward(l1, l2, l3);
  // Level0 outputs == model final outputs 
  outputL0_->Forward(l2, s2);
}

void PyNET::Split(vector<gpu::Memory *> inputs) {
  size_t size = inputs[0]->Size() / 2;
  int n = size / sizeof(DATA_TYPE);

  cl_mem inMem = inputs[0]->Ptr();
  cl_mem outMem = inputs[1]->Ptr();

  cl_int err = 0;
  err = clSetKernelArg(splitKernel_, 0, sizeof(cl_mem), &inMem);
  err = clSetKernelArg(splitKernel_, 1, sizeof(cl_mem), &outMem);
  err = clSetKernelArg(splitKernel_, 2, sizeof(cl_int), &n);
  CheckCl(err);

  size_t globalSize[] = {(size_t)n};
  size_t localSize[] = {(size_t)256};

  globalSize[0] =
      (globalSize[0] + localSize[0] - 1) / localSize[0] * localSize[0];

  CheckCl(clEnqueueNDRangeKernel(dev_->ClDnnHandle()->queue, splitKernel_, 1,
                                 NULL, globalSize, localSize, 0, NULL, NULL));

  inputs[0]->SetUsedSize(size);
  inputs[1]->SetUsedSize(size);
}

PyNET::~PyNET() {
  delete convL1D1_;
  delete pool1_;
  delete convL2D1_;
  delete pool2_;
  delete convL3D1_;
  delete pool3_;
  delete convL4D1_;
  delete pool4_;

  delete convL5D1_;
  delete convL5D2_;
  delete convL5D3_;
  delete convL5D4_;
  delete convT4a_;
  delete convT4b_;
  delete convL5Out_;
  delete outputL5_;

  delete convL4D2_;
  delete convL4D3_;
  delete convL4D4_;
  delete convL4D5_;
  delete convL4D6_;
  delete convL4D7_;
  delete convL4D8_;
  delete convT3a_;
  delete convT3b_;
  delete convL4Out_;
  delete outputL4_;

  delete convL3D2_;
  delete convL3D3_;
  delete convL3D4_;
  delete convL3D5_;
  delete convL3D6_;
  delete convL3D7_1_;
  delete convL3D7_;
  delete convL3D8_;
  delete convT2a_;
  delete convT2b_;
  delete convL3Out_;
  delete outputL3_;

  delete convL2D2_;
  delete convL2D3_;
  delete convL2D4_;
  delete convL2D5_;
  delete convL2D6_;
  delete convL2D7_;
  delete convL2D8_;
  delete convL2D9_;
  delete convL2D10_;
  delete convL2D11_;
  delete convL2D12_;
  delete convT1a_;
  delete convT1b_;
  delete convL2Out_;
  delete outputL2_;

  delete convL1D2_;
  delete convL1D3_;
  delete convL1D4_;
  delete convL1D5_;
  delete convL1D6_;
  delete convL1D7_;
  delete convL1D8_;
  delete convL1D9_;
  delete convL1D10_;
  delete convL1D11_;
  delete convL1D12_;
  delete convL1D13_1_;
  delete convL1D13_;
  delete convL1D14_;
  delete convT0_;
  delete convL1Out_;
  delete outputL1_;

  delete convL0D1_;
  delete outputL0_;

  delete convL1D1_H;
  delete convTb_H;
}

}  // namespace op
}  // namespace gpu
