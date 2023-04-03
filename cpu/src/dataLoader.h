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

cpu::Memory *ExtractBayerChannel(cpu::Memory *in, int imageWidth,
                                 int imageHeight);

class DataLoader {
 public:
  // void load_data(string dataset_dir, int dataset_size, float dslr_scale, bool
  // test=false);
  DataLoader(cpu::Device *dev, string dataDir, int dataSize, float scale,
             int level, bool fullResolution);
  ~DataLoader();
  void ExtractBayerChannel(cpu::Memory *input, cpu::Memory *output);
  int Size();
  void Get(int i, cpu::Memory *l1, cpu::Memory *l2);
  void GetDataSize(int *c, int *h, int *w);

 private:
  PngLoader pngLoader;
  vector<string> filesPath;

  int dataSize_;
  cpu::Device *dev_;

  int inputChannel_;
  int inputHeight_;
  int inputWidth_;

  int outputChannel_;
  int outputHeight_;
  int outputWidth_;
};

#endif
