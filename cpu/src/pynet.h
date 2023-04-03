#ifndef PYNET_H_
#define PYNET_H_

#include <fstream>
#include <iostream>

#include "cpu.h"
#include "operation.h"

using namespace std;

namespace cpu {
namespace op {

class PyNET : public cpu::op::Operation {
 public:
  PyNET(cpu::Device* dev, int inputChannel, int inputHeight, int inputWidth,
        int level, bool instanceOn = true, bool instanceOnL1 = false);
  virtual ~PyNET();

  void LoadWeight(char* weightFile);
  void Forward(cpu::Memory* _l1, cpu::Memory* _l2, cpu::Memory* _l3,
               cpu::Memory* _s1, cpu::Memory* _s2);
  void Add(cpu::Memory* in1, cpu::Memory* in2);

  void level5();
  void level4();
  void level3();
  void level2();
  void level1();
  void level0();

 private:
  int level_;
  bool instanceOn_;
  bool instanceOnL1_;

  // layer 5
  cpu::op::ConvMultiBlock* convL5D1_;
  cpu::op::ConvMultiBlock* convL5D2_;
  cpu::op::ConvMultiBlock* convL5D3_;
  cpu::op::ConvMultiBlock* convL5D4_;
  cpu::op::UpsampleConvLayer* convT4a_;
  cpu::op::UpsampleConvLayer* convT4b_;
  cpu::op::ConvLayer* convL5Out_;
  cpu::op::Sigmoid* outputL5_;

  // layer 4
  cpu::op::MaxPool2d* pool4_;
  cpu::op::ConvMultiBlock* convL4D1_;
  cpu::op::Cat* convL4D2_;
  cpu::op::ConvMultiBlock* convL4D3_;
  cpu::op::ConvMultiBlock* convL4D4_;
  cpu::op::ConvMultiBlock* convL4D5_;
  cpu::op::ConvMultiBlock* convL4D6_;
  cpu::op::Cat* convL4D7_;
  cpu::op::ConvMultiBlock* convL4D8_;
  cpu::op::UpsampleConvLayer* convT3a_;
  cpu::op::UpsampleConvLayer* convT3b_;
  cpu::op::ConvLayer* convL4Out_;
  cpu::op::Sigmoid* outputL4_;

  // layer 3
  cpu::op::MaxPool2d* pool3_;
  cpu::op::ConvMultiBlock* convL3D1_;
  cpu::op::Cat* convL3D2_;
  cpu::op::ConvMultiBlock* convL3D3_;
  cpu::op::ConvMultiBlock* convL3D4_;
  cpu::op::ConvMultiBlock* convL3D5_;
  cpu::op::ConvMultiBlock* convL3D6_;
  cpu::op::Cat* convL3D7_1_;
  cpu::op::Cat* convL3D7_;
  cpu::op::ConvMultiBlock* convL3D8_;
  cpu::op::UpsampleConvLayer* convT2a_;
  cpu::op::UpsampleConvLayer* convT2b_;
  cpu::op::ConvLayer* convL3Out_;
  cpu::op::Sigmoid* outputL3_;

  // layer 2
  cpu::op::MaxPool2d* pool2_;
  cpu::op::ConvMultiBlock* convL2D1_;
  cpu::op::Cat* convL2D2_;
  cpu::op::ConvMultiBlock* convL2D3_;
  cpu::op::Cat* convL2D4_;
  cpu::op::ConvMultiBlock* convL2D5_;
  cpu::op::ConvMultiBlock* convL2D6_;
  cpu::op::ConvMultiBlock* convL2D7_;
  cpu::op::ConvMultiBlock* convL2D8_;
  cpu::op::Cat* convL2D9_;
  cpu::op::ConvMultiBlock* convL2D10_;
  cpu::op::Cat* convL2D11_;
  cpu::op::ConvMultiBlock* convL2D12_;
  cpu::op::UpsampleConvLayer* convT1a_;
  cpu::op::UpsampleConvLayer* convT1b_;
  cpu::op::ConvLayer* convL2Out_;
  cpu::op::Sigmoid* outputL2_;

  // layer 1
  cpu::op::MaxPool2d* pool1_;
  cpu::op::ConvMultiBlock* convL1D1_;
  cpu::op::Cat* convL1D2_;
  cpu::op::ConvMultiBlock* convL1D3_;
  cpu::op::Cat* convL1D4_;
  cpu::op::ConvMultiBlock* convL1D5_;
  cpu::op::ConvMultiBlock* convL1D6_;
  cpu::op::ConvMultiBlock* convL1D7_;
  cpu::op::ConvMultiBlock* convL1D8_;
  cpu::op::ConvMultiBlock* convL1D9_;
  cpu::op::ConvMultiBlock* convL1D10_;
  cpu::op::Cat* convL1D11_;
  cpu::op::ConvMultiBlock* convL1D12_;
  cpu::op::Cat* convL1D13_1_;
  cpu::op::Cat* convL1D13_;
  cpu::op::ConvMultiBlock* convL1D14_;
  cpu::op::UpsampleConvLayer* convT0_;
  cpu::op::ConvLayer* convL1Out_;
  cpu::op::Sigmoid* outputL1_;

  // level 0 : final layer
  cpu::op::ConvLayer* convL0D1_;
  cpu::op::Sigmoid* outputL0_;

  cpu::Memory *s1, *s2, *l1, *l2, *l3;

  float* convL4D1_H;
  float* convL3D1_H;
  float* convL2D1_H;
  float* convL1D1_H;
  float* convTb_H;

  size_t convL4D1_Size, convL3D1_Size, convL2D1_Size, convL1D1_Size,
      convTb_Size;
};

}  // namespace op
}  // namespace cpu

#endif
