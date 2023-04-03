#ifndef DATALOADER_H_
#define DATALOADER_H_

#include <cstring>
#include <experimental/filesystem>
#include <iostream>
#include <vector>

#include "device.h"
#include "memory.h"
#include "pngLoader.h"

using namespace std;
using std::experimental::filesystem::directory_iterator;

class DataLoader {
 public:

  DataLoader(gpu::Device *dev, string dataDir, int dataSize, float scale,
             int level, bool fullResolution);
  ~DataLoader();
  void ExtractBayerChannel(gpu::Memory *input, gpu::Memory *output);
  int Size();
  void Get(int i, gpu::Memory *s1, gpu::Memory *s2);
  void GetDataSize(int *c, int *h, int *w);
  vector<string> getFile();

 private:
  PngLoader pngLoader;
  vector<string> filesPath;

  int dataSize_;
  gpu::Device *dev_;

  int inputChannel_;
  int inputHeight_;
  int inputWidth_;

  int outputChannel_;
  int outputHeight_;
  int outputWidth_;

  cl_kernel kernel_extract_bayer;
  cl_kernel kernel_convert_to_fp16;
};

#endif
