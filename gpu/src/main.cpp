
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "../include/gpu.h"
#include "half.hpp"

using namespace std;
using half_float::half;
using namespace half_float::literal;

const size_t input_tile_size = 1645608960 / 2;
const size_t output_tile_size = 1645608960 / 2;
const size_t filter_tile_size = 37748736 / 2;

size_t largeSize = 738410496 / 2;
size_t smallSize = 369205248 / 2;


const size_t groupSum_mem_size = 357120 * 4;
const size_t std_mem_size = 2048 * 4;
const size_t mean_mem_size = 2048 * 4;

cl_mem input_tile, filter_tile, output_tile;
cl_mem stdMem, groupSumMem, meanMem;

void init(gpu::Device *dev_) {
  cl_int err = 0;

  input_tile = clCreateBuffer(dev_->ClDnnHandle()->context, CL_MEM_READ_WRITE,
                              input_tile_size, NULL, &err);
  CheckCl(err);
  filter_tile = clCreateBuffer(dev_->ClDnnHandle()->context, CL_MEM_READ_WRITE,
                               filter_tile_size, NULL, &err);
  CheckCl(err);
  output_tile = clCreateBuffer(dev_->ClDnnHandle()->context, CL_MEM_READ_WRITE,
                               output_tile_size, NULL, &err);
  CheckCl(err);

  stdMem = clCreateBuffer(dev_->ClDnnHandle()->context, CL_MEM_READ_WRITE,
                          std_mem_size, NULL, &err);
  CheckCl(err);
  groupSumMem = clCreateBuffer(dev_->ClDnnHandle()->context, CL_MEM_READ_WRITE,
                               groupSum_mem_size, NULL, &err);
  CheckCl(err);
  meanMem = clCreateBuffer(dev_->ClDnnHandle()->context, CL_MEM_READ_WRITE,
                           mean_mem_size, NULL, &err);
  CheckCl(err);
}

void clean() {
  CheckCl(clReleaseMemObject(input_tile));
  CheckCl(clReleaseMemObject(filter_tile));
  CheckCl(clReleaseMemObject(output_tile));

  CheckCl(clReleaseMemObject(groupSumMem));
  CheckCl(clReleaseMemObject(meanMem));
  CheckCl(clReleaseMemObject(stdMem));
}

int main(int argc, char *argv[]) {
  string datasetDir = argv[1];
  int size = atoi(argv[3]);
  float scale = 1;
  int level = 0;
  bool fullResolution = true;

  gpu::Device *dev = new gpu::Device();
  init(dev);

  cout << "level : " << level << endl;

  ////////////////////////// create dataloader
  int C, H, W;
  DataLoader visualData(dev, datasetDir, size, scale, level, fullResolution);
  cout << "visual dataset count : " << visualData.Size() << endl;
  visualData.GetDataSize(&C, &H, &W);

  gpu::op::PyNET pynet(dev, C, H, W, level, true, true);

  ////////////////////// load weight
  pynet.LoadWeight(argv[2]);

  // char * output = (char *)malloc(largeSize);
  gpu::Memory *l1, *l2, *l3, *s1, *s2;
  ////////////// create buffer
  l1 = new gpu::Memory(dev);
  l2 = new gpu::Memory(dev);
  l3 = new gpu::Memory(dev);
  s1 = new gpu::Memory(dev);
  s2 = new gpu::Memory(dev);

  for (int i = 0; i < visualData.Size(); i++) {
    ////////////// create buffer
    l1->CreateBuffer(largeSize);
    l2->CreateBuffer(largeSize);
    l3->CreateBuffer(largeSize);
    s1->CreateBuffer(smallSize);
    s2->CreateBuffer(smallSize);

    ////////////// load image
    timer_start(0);
    visualData.Get(i, l1, l2);
    pynet.Forward(l2, l1, l3, s1, &s2);


    int h, w, c;
    pynet.GetOutputSize(c, h, w);


    gpu::Memory *_output = new gpu::Memory(dev, c * h * w * sizeof(float));
    float *output = (float *)malloc(c * h * w * sizeof(float));

    ConvertFromFp16(s2, _output, c, h, w);


    size_t output_size = _output->Size();

    _output->Get(output);
    clFinish(dev->ClDnnHandle()->queue);
    cout << "inference time : " << timer_stop(0) << " sec" << endl;

    delete _output;


    ////////////// save png
    cout << "pynet output size : " << c << ", " << h << ", " << w << endl;
    timer_start(0);
    SavePng((char *)output, output_size, i, h, w);
    clFinish(dev->ClDnnHandle()->queue);
    cout << "save png time : " << timer_stop(0) << " sec" << endl;

    clFinish(dev->ClDnnHandle()->queue);
    s2->Release();

    cout << i << " iter" << endl;

    delete output;
  }
  clean();
}
