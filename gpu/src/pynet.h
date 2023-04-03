#ifndef PYNET_H_
#define PYNET_H_

#include <fstream>
#include <iostream>
#include <vector>

#include "gpu.h"
#include "operation.h"

using namespace std;

extern size_t largeSize, smallSize, smallSize2;

namespace gpu {
namespace op {

class PyNET : public gpu::op::Operation {
 public:
  PyNET(gpu::Device* dev, int inputChannel, int inputHeight, int inputWidth,
        int level, bool instanceOn = true, bool instanceOnL1 = false);
  virtual ~PyNET();

  void LoadWeight(char* weightFile);
  void Forward(gpu::Memory* _l1, gpu::Memory* _l2, gpu::Memory* _l3,
               gpu::Memory* _s1, gpu::Memory** _s2);

  void level5();
  void level4();
  void level3();
  void level2();
  void level1();
  void level0();

  void Split(vector<gpu::Memory*> inputs);

 private:
  int level_;
  bool instanceOn_;
  bool instanceOnL1_;

  // layer 5
  gpu::op::ConvMultiBlock* convL5D1_;
  gpu::op::ConvMultiBlock* convL5D2_;
  gpu::op::ConvMultiBlock* convL5D3_;
  gpu::op::ConvMultiBlock* convL5D4_;
  gpu::op::UpsampleConvLayer* convT4a_;
  gpu::op::UpsampleConvLayer* convT4b_;
  gpu::op::ConvLayer* convL5Out_;
  gpu::op::Sigmoid* outputL5_;

  // layer 4
  gpu::op::MaxPool2d* pool4_;
  gpu::op::ConvMultiBlock* convL4D1_;
  gpu::op::Cat* convL4D2_;
  gpu::op::ConvMultiBlock* convL4D3_;
  gpu::op::ConvMultiBlock* convL4D4_;
  gpu::op::ConvMultiBlock* convL4D5_;
  gpu::op::ConvMultiBlock* convL4D6_;
  gpu::op::Cat* convL4D7_;
  gpu::op::ConvMultiBlock* convL4D8_;
  gpu::op::UpsampleConvLayer* convT3a_;
  gpu::op::UpsampleConvLayer* convT3b_;
  gpu::op::ConvLayer* convL4Out_;
  gpu::op::Sigmoid* outputL4_;

  // layer 3
  gpu::op::MaxPool2d* pool3_;
  gpu::op::ConvMultiBlock* convL3D1_;
  gpu::op::Cat* convL3D2_;
  gpu::op::ConvMultiBlock* convL3D3_;
  gpu::op::ConvMultiBlock* convL3D4_;
  gpu::op::ConvMultiBlock* convL3D5_;
  gpu::op::ConvMultiBlock* convL3D6_;
  gpu::op::Cat* convL3D7_1_;
  gpu::op::Cat* convL3D7_;
  gpu::op::ConvMultiBlock* convL3D8_;
  gpu::op::UpsampleConvLayer* convT2a_;
  gpu::op::UpsampleConvLayer* convT2b_;
  gpu::op::ConvLayer* convL3Out_;
  gpu::op::Sigmoid* outputL3_;

  // layer 2
  gpu::op::MaxPool2d* pool2_;
  gpu::op::ConvMultiBlock* convL2D1_;
  gpu::op::Cat* convL2D2_;
  gpu::op::ConvMultiBlock* convL2D3_;
  gpu::op::Cat* convL2D4_;
  gpu::op::ConvMultiBlock* convL2D5_;
  gpu::op::ConvMultiBlock* convL2D6_;
  gpu::op::ConvMultiBlock* convL2D7_;
  gpu::op::ConvMultiBlock* convL2D8_;
  gpu::op::Cat* convL2D9_;
  gpu::op::ConvMultiBlock* convL2D10_;
  gpu::op::Cat* convL2D11_;
  gpu::op::ConvMultiBlock* convL2D12_;
  gpu::op::UpsampleConvLayer* convT1a_;
  gpu::op::UpsampleConvLayer* convT1b_;
  gpu::op::ConvLayer* convL2Out_;
  gpu::op::Sigmoid* outputL2_;

  // layer 1
  gpu::op::MaxPool2d* pool1_;
  gpu::op::ConvMultiBlock* convL1D1_;
  gpu::op::Cat* convL1D2_;
  gpu::op::ConvMultiBlock2* convL1D3_;
  gpu::op::Cat* convL1D4_;
  gpu::op::ConvMultiBlock2* convL1D5_;
  gpu::op::ConvMultiBlock2* convL1D6_;
  gpu::op::ConvMultiBlock2* convL1D7_;
  gpu::op::ConvMultiBlock2* convL1D8_;
  gpu::op::ConvMultiBlock2* convL1D9_;
  gpu::op::ConvMultiBlock2* convL1D10_;
  gpu::op::Cat* convL1D11_;
  gpu::op::ConvMultiBlock2* convL1D12_;
  gpu::op::Cat* convL1D13_1_;
  gpu::op::Cat* convL1D13_;
  gpu::op::ConvMultiBlock2* convL1D14_;
  gpu::op::UpsampleConvLayer2* convT0_;
  gpu::op::ConvLayer* convL1Out_;
  gpu::op::Sigmoid* outputL1_;

  // level 0 : final layer
  gpu::op::ConvLayer* convL0D1_;
  gpu::op::Sigmoid* outputL0_;

  // add kernel
  cl_kernel splitKernel_;

  gpu::Memory *l1, *l2, *l3, *s1, *s2;
  vector<gpu::Memory*> inputs;
  vector<gpu::Memory*> outputs;

  gpu::Memory *convL2D1, *convL3D1, *convL4D1, *convTb;

  // host memeory
  DATA_TYPE* convL1D1_H;
  DATA_TYPE* convTb_H;

  size_t convL4D1_Size, convL3D1_Size, convL2D1_Size, convL1D1_Size, convT_Size;
};

}  // namespace op
}  // namespace gpu

#endif
