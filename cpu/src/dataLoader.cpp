
#include "dataLoader.h"

#include "util.h"

void DataLoader::ExtractBayerChannel(cpu::Memory *input, cpu::Memory *output) {
  // (imageHeight, imageWidth) --> (4, imageHeight/2, imageWidth/2)

  // extract bayer
  size_t *I = (size_t *)input->Ptr();
  float *O = (float *)output->Ptr();

  // channel order : B -> GB -> R -> GR
  float norm = 4 * 255;

  for (int ih = 0; ih < inputHeight_; ih++) {
    for (int iw = 0; iw < inputWidth_; iw++) {
      int idx = ih * inputWidth_ + iw;
      int oh = ih / 2;
      int ow = iw / 2;

      if ((ih & 0x1) == 0) {
        if ((iw & 0x1) == 0) {
          // R channel
          O[(2 * outputHeight_ + oh) * outputWidth_ + ow] =
              (float)I[idx] / norm;
        } else {
          // GB channel
          O[(outputHeight_ + oh) * outputWidth_ + ow] = (float)I[idx] / norm;
        }
      } else {
        if ((iw & 0x1) == 0) {
          // GR channel
          O[(3 * outputHeight_ + oh) * outputWidth_ + ow] =
              (float)I[idx] / norm;
        } else {
          // B channel
          O[oh * outputWidth_ + ow] = (float)I[idx] / norm;
        }
      }
    }
  }

  output->SetUsedSize(outputChannel_ * outputHeight_ * outputWidth_ *
                      sizeof(float));
}

DataLoader::DataLoader(cpu::Device *dev, string dataDir, int dataSize,
                       float scale, int level, bool fullResolution) {
  dev_ = dev;
  dataSize_ = dataSize;
  string rawDir = dataDir + "/" + "test" + "/" + "huawei_full_resolution";

  for (const auto &file : directory_iterator(rawDir)) {
    cout << file.path() << endl;
    filesPath.push_back(file.path());
  }

  inputChannel_ = 1;
  inputHeight_ = 2880;
  inputWidth_ = 3968;

  outputChannel_ = 4;
  outputHeight_ = inputHeight_ / 2;
  outputWidth_ = inputWidth_ / 2;
}

DataLoader::~DataLoader() {}

int DataLoader::Size() { return dataSize_; }

void DataLoader::Get(int i, cpu::Memory *l1, cpu::Memory *l2) {
  char filePath_[500];
  strcpy(filePath_, filesPath[i].c_str());
  pngLoader.ReadFile(filePath_);
  size_t *ptr = pngLoader.Ptr();

  l1->SetUsedSize(pngLoader.Size());
  l1->Set(ptr);
  delete ptr;

  ExtractBayerChannel(l1, l2);
}

void DataLoader::GetDataSize(int *c, int *h, int *w) {
  *c = outputChannel_;
  *h = outputHeight_;
  *w = outputWidth_;
}
