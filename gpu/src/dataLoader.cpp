#include <dataLoader.h>
#include <util.h>

void DataLoader::ExtractBayerChannel(gpu::Memory *input, gpu::Memory *output) {
  cl_mem inputMem = input->Ptr();
  cl_mem outputMem = output->Ptr();

  cl_int err = 0;
  err = clSetKernelArg(kernel_extract_bayer, 0, sizeof(cl_mem), &inputMem);
  err = clSetKernelArg(kernel_extract_bayer, 1, sizeof(cl_mem), &outputMem);
  err = clSetKernelArg(kernel_extract_bayer, 2, sizeof(cl_int), &inputChannel_);
  err = clSetKernelArg(kernel_extract_bayer, 3, sizeof(cl_int), &inputHeight_);
  err = clSetKernelArg(kernel_extract_bayer, 4, sizeof(cl_int), &inputWidth_);
  err =
      clSetKernelArg(kernel_extract_bayer, 5, sizeof(cl_int), &outputChannel_);
  err = clSetKernelArg(kernel_extract_bayer, 6, sizeof(cl_int), &outputHeight_);
  err = clSetKernelArg(kernel_extract_bayer, 7, sizeof(cl_int), &outputWidth_);
  CheckCl(err);

  size_t gws[2] = {(size_t)inputWidth_, (size_t)inputHeight_};
  size_t lws[2] = {16, 16};

  gws[0] = (gws[0] + lws[0] - 1) / lws[0] * lws[0];
  gws[1] = (gws[1] + lws[1] - 1) / lws[1] * lws[1];

  CheckCl(clEnqueueNDRangeKernel(dev_->ClDnnHandle()->queue,
                                 kernel_extract_bayer, 2, NULL, gws, lws, 0,
                                 NULL, NULL));

  output->SetUsedSize(outputChannel_ * outputHeight_ * outputWidth_ *
                      sizeof(DATA_TYPE));
}

DataLoader::DataLoader(gpu::Device *dev, string dataDir, int dataSize,
                       float scale, int level, bool fullResolution) {
  dev_ = dev;
  dataSize_ = dataSize;
  string rawDir = dataDir + "/" + "test" + "/" + "huawei_full_resolution";

  for (const auto &file : directory_iterator(rawDir)) {
    cout << file.path() << endl;
    filesPath.push_back(file.path());
  }

  // 실제 입력 파일의 갯수보다 많이 추론을 명령하는 경우.
  if (dataSize_ > filesPath.size()) dataSize_ = filesPath.size();

  inputChannel_ = 1;
  inputHeight_ = 2880;
  inputWidth_ = 3968;

  outputChannel_ = 4;
  outputHeight_ = inputHeight_ / 2;
  outputWidth_ = inputWidth_ / 2;

  cl_int err = 0;
  kernel_extract_bayer =
      clCreateKernel(dev_->ClDnnHandle()->program, "extractBayerChannel", &err);
  CheckCl(err);
}

DataLoader::~DataLoader() {}

int DataLoader::Size() { return dataSize_; }

void DataLoader::Get(int i, gpu::Memory *s1, gpu::Memory *s2) {
  char filePath_[500];
  strcpy(filePath_, filesPath[i].c_str());
  pngLoader.ReadFile(filePath_);

  int size = pngLoader.Size();

  cl_int status;
  size_t *ptr = (size_t *)clEnqueueMapBuffer(dev_->ClDnnHandle()->queue,
                                             s1->Ptr(), CL_TRUE, CL_MAP_WRITE,
                                             0, size, 0, NULL, NULL, &status);
  pngLoader.SetPtr(&ptr);

  // cpu->gpu
  s1->SetUsedSize(size);
  clEnqueueUnmapMemObject(dev_->ClDnnHandle()->queue, s1->Ptr(), (void *)ptr, 0,
                          NULL, NULL);

  clFinish(dev_->ClDnnHandle()->queue);

  ExtractBayerChannel(s1, s2);
}

void DataLoader::GetDataSize(int *c, int *h, int *w) {
  *c = outputChannel_;
  *h = outputHeight_;
  *w = outputWidth_;
}
