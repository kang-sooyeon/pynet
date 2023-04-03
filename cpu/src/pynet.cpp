#include <pynet.h>

/*
#define DEBUG_OUT 1
#define DEBUG_L5 1
#define DEBUG_L4 1
#define DEBUG_L3 1
#define DEBUG_L2 1
#define DEBUG_L1 1
#define DEBUG_L0 1
*/

extern size_t smallSize;

namespace cpu {
namespace op {
PyNET::PyNET(cpu::Device *dev, int inputChannel, int inputHeight,
             int inputWidth, int level, bool instanceOn, bool instanceOnL1)
    : cpu::op::Operation(dev, inputChannel, inputHeight, inputWidth),
      level_(level),
      instanceOn_(instanceOn),
      instanceOnL1_(instanceOnL1) {
  int c, h, w, c2, h2, w2;

  // D1
  convL1D1_ = new cpu::op::ConvMultiBlock(dev, 4, inputHeight_, inputWidth_, 32,
                                          3, false);
  convL1D1_->GetOutputSize(c, h, w);
  convL1D1_Size = c * h * w * sizeof(float);

  pool1_ = new cpu::op::MaxPool2d(dev, c, h, w, 2, 2);
  pool1_->GetOutputSize(c, h, w);

  convL2D1_ = new cpu::op::ConvMultiBlock(dev, 32, h, w, 64, 3, instanceOn_);
  convL2D1_->GetOutputSize(c, h, w);
  convL2D1_Size = c * h * w * sizeof(float);
  pool2_ = new cpu::op::MaxPool2d(dev, c, h, w, 2, 2);
  pool2_->GetOutputSize(c, h, w);

  convL3D1_ = new cpu::op::ConvMultiBlock(dev, 64, h, w, 128, 3, instanceOn);
  convL3D1_->GetOutputSize(c, h, w);
  convL3D1_Size = c * h * w * sizeof(float);
  pool3_ = new cpu::op::MaxPool2d(dev, c, h, w, 2, 2);
  pool3_->GetOutputSize(c, h, w);

  convL4D1_ = new cpu::op::ConvMultiBlock(dev, 128, h, w, 256, 3, instanceOn);
  convL4D1_->GetOutputSize(c, h, w);
  convL4D1_Size = c * h * w * sizeof(float);
  pool4_ = new cpu::op::MaxPool2d(dev, c, h, w, 2, 2);
  pool4_->GetOutputSize(c, h, w);

  // level 5
  convL5D1_ = new cpu::op::ConvMultiBlock(dev, 256, h, w, 512, 3, instanceOn);
  convL5D1_->GetOutputSize(c, h, w);
  convL5D2_ = new cpu::op::ConvMultiBlock(dev, 512, h, w, 512, 3, instanceOn);
  convL5D2_->GetOutputSize(c, h, w);
  convL5D3_ = new cpu::op::ConvMultiBlock(dev, 512, h, w, 512, 3, instanceOn);
  convL5D3_->GetOutputSize(c, h, w);
  convL5D4_ = new cpu::op::ConvMultiBlock(dev, 512, h, w, 512, 3, instanceOn);
  convL5D4_->GetOutputSize(c, h, w);
  convT4a_ = new cpu::op::UpsampleConvLayer(dev, 512, h, w, 256, 3);
  convT4b_ = new cpu::op::UpsampleConvLayer(dev, 512, h, w, 256, 3);
  convL5Out_ = new cpu::op::ConvLayer(dev, 512, h, w, 3, 3, 1, false, false);
  convL5Out_->GetOutputSize(c, h, w);
  outputL5_ = new cpu::op::Sigmoid(dev, c, h, w);

  // level 4
  convL4D1_->GetOutputSize(c, h, w);
  convT4a_->GetOutputSize(c2, h2, w2);
  convL4D2_ = new cpu::op::Cat(dev, c, h, w, c2, h2, w2, 1);
  convL4D2_->GetOutputSize(c, h, w);
  convL4D3_ = new cpu::op::ConvMultiBlock(dev, 512, h, w, 256, 3, instanceOn);
  convL4D3_->GetOutputSize(c, h, w);
  convL4D4_ = new cpu::op::ConvMultiBlock(dev, 256, h, w, 256, 3, instanceOn);
  convL4D4_->GetOutputSize(c, h, w);
  convL4D5_ = new cpu::op::ConvMultiBlock(dev, 256, h, w, 256, 3, instanceOn);
  convL4D5_->GetOutputSize(c, h, w);
  convL4D6_ = new cpu::op::ConvMultiBlock(dev, 256, h, w, 256, 3, instanceOn);
  convL4D6_->GetOutputSize(c, h, w);
  convT4b_->GetOutputSize(c2, h2, w2);
  convL4D7_ = new cpu::op::Cat(dev, c, h, w, c2, h2, w2, 1);
  convL4D7_->GetOutputSize(c, h, w);
  convL4D8_ = new cpu::op::ConvMultiBlock(dev, 512, h, w, 256, 3, instanceOn);
  convL4D8_->GetOutputSize(c, h, w);
  convT3a_ = new cpu::op::UpsampleConvLayer(dev, 256, h, w, 128, 3);
  convT3b_ = new cpu::op::UpsampleConvLayer(dev, 256, h, w, 128, 3);
  convL4Out_ = new cpu::op::ConvLayer(dev, 256, h, w, 3, 3, 1, false, false);
  convL4Out_->GetOutputSize(c, h, w);
  outputL4_ = new cpu::op::Sigmoid(dev, c, h, w);

  // level 3
  convL3D1_->GetOutputSize(c, h, w);
  convT3a_->GetOutputSize(c2, h2, w2);
  convL3D2_ = new cpu::op::Cat(dev, c, h, w, c2, h2, w2, 1);
  convL3D2_->GetOutputSize(c, h, w);
  convL3D3_ = new cpu::op::ConvMultiBlock(dev, 256, h, w, 128, 5, instanceOn);
  convL3D3_->GetOutputSize(c, h, w);
  convL3D4_ = new cpu::op::ConvMultiBlock(dev, 256, h, w, 128, 5, instanceOn);
  convL3D4_->GetOutputSize(c, h, w);
  convL3D5_ = new cpu::op::ConvMultiBlock(dev, 256, h, w, 128, 5, instanceOn);
  convL3D5_->GetOutputSize(c, h, w);
  convL3D6_ = new cpu::op::ConvMultiBlock(dev, 256, h, w, 128, 5, instanceOn);

  convL3D6_->GetOutputSize(c, h, w);
  convL3D1_->GetOutputSize(c2, h2, w2);
  convL3D7_1_ = new cpu::op::Cat(dev, c, h, w, c2, h2, w2, 1);
  convL3D7_1_->GetOutputSize(c, h, w);
  convT3b_->GetOutputSize(c2, h2, w2);
  convL3D7_ = new cpu::op::Cat(dev, c, h, w, c2, h2, w2, 1);

  convL3D7_->GetOutputSize(c, h, w);
  convL3D8_ = new cpu::op::ConvMultiBlock(dev, 512, h, w, 128, 3, instanceOn);
  convL3D8_->GetOutputSize(c, h, w);
  convT2a_ = new cpu::op::UpsampleConvLayer(dev, 128, h, w, 64, 3);
  convT2b_ = new cpu::op::UpsampleConvLayer(dev, 128, h, w, 64, 3);
  convL3Out_ = new cpu::op::ConvLayer(dev, 128, h, w, 3, 3, 1, false, false);
  convL3Out_->GetOutputSize(c, h, w);
  outputL3_ = new cpu::op::Sigmoid(dev, c, h, w);

  // level 2
  convL2D1_->GetOutputSize(c, h, w);
  convT2a_->GetOutputSize(c2, h2, w2);
  convL2D2_ = new cpu::op::Cat(dev, c, h, w, c2, h2, w2, 1);
  convL2D2_->GetOutputSize(c, h, w);
  convL2D3_ = new cpu::op::ConvMultiBlock(dev, 128, h, w, 64, 5, instanceOn);
  convL2D3_->GetOutputSize(c, h, w);
  convL2D1_->GetOutputSize(c2, h2, w2);
  convL2D4_ = new cpu::op::Cat(dev, c, h, w, c2, h2, w2, 1);
  convL2D4_->GetOutputSize(c, h, w);
  convL2D5_ = new cpu::op::ConvMultiBlock(dev, 192, h, w, 64, 7, instanceOn);
  convL2D5_->GetOutputSize(c, h, w);
  convL2D6_ = new cpu::op::ConvMultiBlock(dev, 192, h, w, 64, 7, instanceOn);
  convL2D6_->GetOutputSize(c, h, w);
  convL2D7_ = new cpu::op::ConvMultiBlock(dev, 192, h, w, 64, 7, instanceOn);
  convL2D7_->GetOutputSize(c, h, w);
  convL2D8_ = new cpu::op::ConvMultiBlock(dev, 192, h, w, 64, 7, instanceOn);
  convL2D8_->GetOutputSize(c, h, w);
  convL2D1_->GetOutputSize(c2, h2, w2);
  convL2D9_ = new cpu::op::Cat(dev, c, h, w, c2, h2, w2, 1);
  convL2D9_->GetOutputSize(c, h, w);
  convL2D10_ = new cpu::op::ConvMultiBlock(dev, 256, h, w, 64, 5, instanceOn);
  convL2D10_->GetOutputSize(c, h, w);
  convT2b_->GetOutputSize(c2, h2, w2);
  convL2D11_ = new cpu::op::Cat(dev, c, h, w, c2, h2, w2, 1);
  convL2D11_->GetOutputSize(c, h, w);
  convL2D12_ = new cpu::op::ConvMultiBlock(dev, 192, h, w, 64, 3, instanceOn);
  convL2D12_->GetOutputSize(c, h, w);
  convT1a_ = new cpu::op::UpsampleConvLayer(dev, 64, h, w, 32, 3);
  convT1b_ = new cpu::op::UpsampleConvLayer(dev, 64, h, w, 32, 3);
  convL2Out_ = new cpu::op::ConvLayer(dev, 64, h, w, 3, 3, 1, false, false);
  convL2Out_->GetOutputSize(c, h, w);
  outputL2_ = new cpu::op::Sigmoid(dev, c, h, w);

  // level 1
  convL1D1_->GetOutputSize(c, h, w);
  convT1a_->GetOutputSize(c2, h2, w2);
  convL1D2_ = new cpu::op::Cat(dev, c, h, w, c2, h2, w2, 1);
  convL1D2_->GetOutputSize(c, h, w);
  convL1D3_ = new cpu::op::ConvMultiBlock(dev, 64, h, w, 32, 5, false);
  convL1D3_->GetOutputSize(c, h, w);
  convL1D1_->GetOutputSize(c2, h2, w2);
  convL1D4_ = new cpu::op::Cat(dev, c, h, w, c2, h2, w2, 1);
  convL1D4_->GetOutputSize(c, h, w);

  convL1D5_ = new cpu::op::ConvMultiBlock(dev, 96, h, w, 32, 7, instanceOnL1);
  convL1D5_->GetOutputSize(c, h, w);

  convL1D6_ = new cpu::op::ConvMultiBlock(dev, 96, h, w, 32, 9, instanceOnL1);
  convL1D6_->GetOutputSize(c, h, w);
  convL1D7_ = new cpu::op::ConvMultiBlock(dev, 128, h, w, 32, 9, instanceOnL1);
  convL1D7_->GetOutputSize(c, h, w);
  convL1D8_ = new cpu::op::ConvMultiBlock(dev, 128, h, w, 32, 9, instanceOnL1);
  convL1D8_->GetOutputSize(c, h, w);
  convL1D9_ = new cpu::op::ConvMultiBlock(dev, 128, h, w, 32, 9, instanceOnL1);
  convL1D9_->GetOutputSize(c, h, w);

  convL1D10_ = new cpu::op::ConvMultiBlock(dev, 128, h, w, 32, 7, instanceOnL1);
  convL1D10_->GetOutputSize(c, h, w);
  convL1D1_->GetOutputSize(c2, h2, w2);
  convL1D11_ = new cpu::op::Cat(dev, c, h, w, c2, h2, w2, 1);
  convL1D11_->GetOutputSize(c, h, w);
  convL1D12_ = new cpu::op::ConvMultiBlock(dev, 128, h, w, 32, 5, instanceOnL1);

  convL1D12_->GetOutputSize(c, h, w);
  convT1b_->GetOutputSize(c2, h2, w2);
  convL1D13_1_ = new cpu::op::Cat(dev, c, h, w, c2, h2, w2, 1);
  convL1D13_1_->GetOutputSize(c, h, w);
  convL1D1_->GetOutputSize(c2, h2, w2);
  convL1D13_ = new cpu::op::Cat(dev, c, h, w, c2, h2, w2, 1);

  convL1D13_->GetOutputSize(c, h, w);
  convL1D14_ = new cpu::op::ConvMultiBlock(dev, 128, h, w, 32, 3, false);
  convL1D14_->GetOutputSize(c, h, w);

  convT0_ = new cpu::op::UpsampleConvLayer(dev, 32, h, w, 16, 3);

  convL1Out_ = new cpu::op::ConvLayer(dev, 32, h, w, 3, 3, 1, false, false);
  convL1Out_->GetOutputSize(c, h, w);
  outputL1_ = new cpu::op::Sigmoid(dev, c, h, w);

  // level 0
  convT0_->GetOutputSize(c, h, w);
  convL0D1_ = new cpu::op::ConvLayer(dev, 16, h, w, 3, 3, 1, false, false);
  convL0D1_->GetOutputSize(c, h, w);
  outputL0_ = new cpu::op::Sigmoid(dev, c, h, w);

  // set pynet output size
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

  convL1D1_H = (float *)malloc(convL1D1_Size);
  convL2D1_H = (float *)malloc(convL2D1_Size);
  convL3D1_H = (float *)malloc(convL3D1_Size);
  convL4D1_H = (float *)malloc(convL4D1_Size);
  convTb_H = (float *)malloc(smallSize);
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

void PyNET::Add(cpu::Memory *in1, cpu::Memory *in2) {
  size_t size = in1->Size();

  if (size != in2->Size()) {
    std::cerr << " Add() size error" << endl;
    abort();
  }

  float *ptr1 = (float *)in1->Ptr();
  float *ptr2 = (float *)in2->Ptr();

  for (size_t i = 0; i < size / sizeof(float); i++) {
    ptr2[i] = ptr1[i] + ptr2[i];
  }
}

PyNET::~PyNET() {
  delete convL4D1_H;
  delete convL3D1_H;
  delete convL2D1_H;
  delete convL1D1_H;
  delete convTb_H;

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
}

void PyNET::Forward(cpu::Memory *_l1, cpu::Memory *_l2, cpu::Memory *_l3,
                    cpu::Memory *_s1, cpu::Memory *_s2) {
  l1 = _l1;
  l2 = _l2;
  l3 = _l3;
  s1 = _s1;
  s2 = _s2;

  convL1D1_->Forward(l1, l2, l3, s1, s2);
#ifdef DEBUG_L5
  Print("convL1D1", l2);
#endif

  l2->Get(convL1D1_H);  //// convL1D1

  pool1_->Forward(l2, l1);
#ifdef DEBUG_L5
  Print("pool1", l1);
#endif

  convL2D1_->Forward(l1, l2, l3, s1, s2);
#ifdef DEBUG_L5
  Print("convL2D1", l2);
#endif
  l2->Get(convL2D1_H);  // convL2D1

  pool2_->Forward(l2, l1);
#ifdef DEBUG_L5
  Print("pool2", l1);
#endif

  convL3D1_->Forward(l1, l2, l3, s1, s2);
#ifdef DEBUG_L5
  Print("convL3D1", l2);
#endif
  l2->Get(convL3D1_H);  // convL3D1

  pool3_->Forward(l2, l1);
#ifdef DEBUG_L5
  Print("pool3", l1);
#endif

  convL4D1_->Forward(l1, l2, l3, s1, s2);
#ifdef DEBUG_L5
  Print("convL4D1", l2);
#endif
  l2->Get(convL4D1_H);  // convL4D1
  pool4_->Forward(l2, l1);
#ifdef DEBUG_L5
  Print("pool4", l1);
#endif

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
}

void PyNET::level5() {
  convL5D1_->Forward(l1, l2, l3, s1, s2);
#ifdef DEBUG_L5
  Print("convL5D1", l2);
#endif

  convL5D2_->Forward(l2, l1, l3, s1, s2);
#ifdef DEBUG_L5
  Print("convL5D2", l1);
#endif

  convL5D3_->Forward(l1, l2, l3, s1, s2);
#ifdef DEBUG_L5
  Print("convL5D3", l2);
#endif
  convL5D4_->Forward(l2, l1, l3, s1, s2);
#ifdef DEBUG_L5
  Print("convL5D4", l1);
#endif

  convT4b_->Forward(l1, l2, l3);
  l2->Get(convTb_H);  // convT4b
  convTb_Size = l2->Size();

#ifdef DEBUG_L5
  Print("convT4b", l2);
#endif

  convT4a_->Forward(l1, l2, l3);
#ifdef DEBUG_L5
  Print("convT4a", l2);
#endif

  convL5Out_->Forward(l1, s1, s2);
#ifdef DEBUG_L5
  Print("convL5Out", s1);
#endif

  outputL5_->Forward(s1, s1);  // **
#ifdef DEBUG_L5
  Print("outputL5", s1);
#endif
}

void PyNET::level4() {
  l1->SetUsedSize(convL4D1_Size);
  l1->Set(convL4D1_H);

  convL4D2_->Forward(l1, l2, l1);
#ifdef DEBUG_L4
  Print("convL4D2", l1);
#endif
  convL4D3_->Forward(l1, l2, l3, s1, s2);
#ifdef DEBUG_L4
  Print("convL4D3", l2);
#endif
  convL4D4_->Forward(l2, l1, l3, s1, s2);
  Add(l1, l2);
#ifdef DEBUG_L4
  Print("convL4D4", l2);
#endif
  convL4D5_->Forward(l2, l1, l3, s1, s2);
  Add(l1, l2);
#ifdef DEBUG_L4
  Print("convL4D5", l2);
#endif

  convL4D6_->Forward(l2, l1, l3, s1, s2);
#ifdef DEBUG_L4
  Print("convL4D6", l1);
#endif

  s2->SetUsedSize(convTb_Size);
  s2->Set(convTb_H);
  convL4D7_->Forward(l1, s2, l1);
#ifdef DEBUG_L4
  Print("convL4D7", l1);
#endif

  convL4D8_->Forward(l1, l2, l3, s1, s2);
#ifdef DEBUG_L4
  Print("convL4D8", l2);
#endif

  convT3b_->Forward(l2, l1, l3);
  l1->Get(convTb_H);
  convTb_Size = l1->Size();
#ifdef DEBUG_L4
  Print("convT3b", l1);
#endif

  convT3a_->Forward(l2, l1, l3);
#ifdef DEBUG_L4
  Print("convT3a", l1);
#endif

  convL4Out_->Forward(l2, s1, s2);
#ifdef DEBUG_L4
  Print("convL4Out", s1);
#endif

  outputL4_->Forward(s1, s1);
#ifdef DEBUG_L4
#ifdef DEBUG_OUT
  Print("outputL4", s1);
#endif
#endif
}

void PyNET::level3() {
  l2->SetUsedSize(convL3D1_Size);
  l2->Set(convL3D1_H);

  convL3D2_->Forward(l2, l1, l2);
#ifdef DEBUG_L3
  Print("convL3D2", l2);
#endif

  convL3D3_->Forward(l2, l1, l3, s1, s2);
  Add(l1, l2);
#ifdef DEBUG_L3
  Print("convL3D3", l2);
#endif
  convL3D4_->Forward(l2, l1, l3, s1, s2);
  Add(l1, l2);
#ifdef DEBUG_L3
  Print("convL3D4", l2);
#endif
  convL3D5_->Forward(l2, l1, l3, s1, s2);
  Add(l1, l2);
#ifdef DEBUG_L3
  Print("convL3D5", l2);
#endif
  convL3D6_->Forward(l2, l1, l3, s1, s2);
#ifdef DEBUG_L3
  Print("convL3D6", l1);
#endif

  s1->SetUsedSize(convL3D1_Size);
  s1->Set(convL3D1_H);
  convL3D7_1_->Forward(l1, s1, l1);
#ifdef DEBUG_L3
  Print("convL3D7_1", l1);
#endif

  s2->SetUsedSize(convTb_Size);
  s2->Set(convTb_H);
  convL3D7_->Forward(l1, s2, l1);
#ifdef DEBUG_L3
  Print("convL3D7", l1);
#endif

  convL3D8_->Forward(l1, l2, l3, s1, s2);
#ifdef DEBUG_L3
  Print("convL3D8", l2);
#endif

  convT2b_->Forward(l2, l1, l3);
  l1->Get(convTb_H);
  convTb_Size = l1->Size();
#ifdef DEBUG_L3
  Print("convT2b", l1);
#endif

  convT2a_->Forward(l2, l1, l3);
#ifdef DEBUG_L3
  Print("convT2a", l1);
#endif

  convL3Out_->Forward(l2, s1, s2);
#ifdef DEBUG_L3
  Print("convL3Out", s1);
#endif
  outputL3_->Forward(s1, s1);
#ifdef DEBUG_L3
#ifdef DEBUG_OUT
  Print("outputL3", s1);
#endif
#endif
}

void PyNET::level2() {
  l2->SetUsedSize(convL2D1_Size);
  l2->Set(convL2D1_H);

  convL2D2_->Forward(l2, l1, l2);
#ifdef DEBUG_L2
  Print("convL2D2", l2);
#endif

  convL2D3_->Forward(l2, l1, l3, s1, s2);
#ifdef DEBUG_L2
  Print("convL2D3", l1);
#endif

  s1->SetUsedSize(convL2D1_Size);
  s1->Set(convL2D1_H);
  convL2D4_->Forward(l1, s1, l1);
#ifdef DEBUG_L2
  Print("convL2D4", l1);
#endif

  convL2D5_->Forward(l1, l2, l3, s1, s2);
  Add(l1, l2);
#ifdef DEBUG_L2
  Print("convL2D5", l2);
#endif

  convL2D6_->Forward(l2, l1, l3, s1, s2);
  Add(l1, l2);
#ifdef DEBUG_L2
  Print("convL2D6", l2);
#endif
  convL2D7_->Forward(l2, l1, l3, s1, s2);
  Add(l1, l2);
#ifdef DEBUG_L2
  Print("convL2D7", l2);
#endif

  convL2D8_->Forward(l2, l1, l3, s1, s2);
#ifdef DEBUG_L2
  Print("convL2D8", l1);
#endif

  s1->SetUsedSize(convL2D1_Size);
  s1->Set(convL2D1_H);
  convL2D9_->Forward(l1, s1, l1);
#ifdef DEBUG_L2
  Print("convL2D9", l1);
#endif

  convL2D10_->Forward(l1, l2, l3, s1, s2);
#ifdef DEBUG_L2
  Print("convL2D10", l2);
#endif

  s2->SetUsedSize(convTb_Size);
  s2->Set(convTb_H);
  convL2D11_->Forward(l2, s2, l2);
#ifdef DEBUG_L2
  Print("convL2D11", l2);
#endif

  convL2D12_->Forward(l2, l1, l3, s1, s2);
#ifdef DEBUG_L2
  Print("convL2D12", l1);
#endif

  convT1b_->Forward(l1, l2, l3);
  l2->Get(convTb_H);
  convTb_Size = l2->Size();
#ifdef DEBUG_L2
  Print("convT1b", l2);
#endif

  convT1a_->Forward(l1, l2, l3);
#ifdef DEBUG_L2
  Print("convT1a", l2);
#endif

  convL2Out_->Forward(l1, s1, s2);
#ifdef DEBUG_L2
  Print("convL2Out", s1);
#endif

  outputL2_->Forward(s1, s1);
#ifdef DEBUG_OUT
  Print("outputL2", s1);
#endif
}

void PyNET::level1() {
  l1->SetUsedSize(convL1D1_Size);
  l1->Set(convL1D1_H);

  convL1D2_->Forward(l1, l2, l1);
#ifdef DEBUG_L1
  Print("convL1D2", l1);
#endif

  convL1D3_->Forward(l1, l2, l3, s1, s2);
#ifdef DEBUG_L1
  Print("convL1D3", l2);
#endif

  s1->SetUsedSize(convL1D1_Size);
  s1->Set(convL1D1_H);
  convL1D4_->Forward(l2, s1, l2);
#ifdef DEBUG_L1
  Print("convL1D4", l2);
#endif

  convL1D5_->Forward(l2, l1, l3, s1, s2);
#ifdef DEBUG_L1
  Print("convL1D5", l1);
#endif

  convL1D6_->Forward(l1, l2, l3, s1, s2);
#ifdef DEBUG_L1
  Print("convL1D6", l2);
#endif

  convL1D7_->Forward(l2, l1, l3, s1, s2);
  Add(l1, l2);
#ifdef DEBUG_L1
  Print("convL1D7", l2);
#endif

  convL1D8_->Forward(l2, l1, l3, s1, s2);

  Add(l1, l2);
#ifdef DEBUG_L1
  Print("convL1D8", l2);
#endif

  convL1D9_->Forward(l2, l1, l3, s1, s2);
  Add(l1, l2);
#ifdef DEBUG_L1
  Print("convL1D9", l2);
#endif

  convL1D10_->Forward(l2, l1, l3, s1, s2);
#ifdef DEBUG_L1
  Print("convL1D10", l1);
#endif

  s1->SetUsedSize(convL1D1_Size);
  s1->Set(convL1D1_H);
  convL1D11_->Forward(l1, s1, l1);
#ifdef DEBUG_L1
  Print("convL1D11", l1);
#endif

  convL1D12_->Forward(l1, l2, l3, s1, s2);
#ifdef DEBUG_L1
  Print("convL1D12", l2);
#endif

  s2->SetUsedSize(convTb_Size);
  s2->Set(convTb_H);
  convL1D13_1_->Forward(l2, s2, l2);
  s1->SetUsedSize(convL1D1_Size);
  s1->Set(convL1D1_H);
  convL1D13_->Forward(l2, s1, l2);
#ifdef DEBUG_L1
  Print("convL1D13", l2);
#endif

  convL1D14_->Forward(l2, l1, l3, s1, s2);
#ifdef DEBUG_L1
  Print("convL1D14", l1);
#endif

  convT0_->Forward(l1, l2, l3);
#ifdef DEBUG_L1
  Print("convT0", l2);
#endif

  convL1Out_->Forward(l1, s1, s2);
#ifdef DEBUG_L1
  Print("convL1Out", s1);
#endif

  outputL1_->Forward(s1, s1);
#ifdef DEBUG_OUT
  Print("outputL1", s1);
#endif
}

void PyNET::level0() {
  convL0D1_->Forward(l2, s1, l1);
#ifdef DEBUG_L0
  Print("convL0D1", s1);
#endif
  outputL0_->Forward(s1, s1);
#ifdef DEBUG_OUT
  Print("outputL0", s1);
#endif
}
}  // namespace op
}  // namespace cpu
