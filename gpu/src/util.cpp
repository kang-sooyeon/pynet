
#include <convolution2d.h>
#include <math.h>
#include <sys/time.h>
#include <util.h>

#include <cstring>
#include <fstream>

#include "svpng.inc"

#define EPSILON 0.01

static double start_time[8];

void Print(char *layerName, gpu::Memory *m) {
  std::cout << "------------------------------------------------------------"
            << std::endl;
  std::cout << layerName << std::endl;
  m->Print();
  std::cout << "Sum : " << m->Sumf() << std::endl;
  std::cout << "Size : " << m->Size() << std::endl;
  std::cout << "------------------------------------------------------------"
            << std::endl;
}

void PrintWithSave(char *layerName, gpu::Memory *m) {
  std::cout << "------------------------------------------------------------"
            << std::endl;
  std::cout << layerName << std::endl;
  m->Print();
  std::cout << "Sum : " << m->Sumf() << std::endl;
  std::cout << "Size : " << m->Size() << std::endl;

  void *buf = malloc(m->Size());
  m->Get(buf);

  char fileName[100];
  sprintf(fileName, "./data/%s.bin", layerName);
  FILE *fp = fopen(fileName, "wb");
  fwrite(buf, m->Size(), 1, fp);
  fclose(fp);

  std::cout << "------------------------------------------------------------"
            << std::endl;
}

void loadData(gpu::Memory *buf, char *fileName) {

  void *tmp = malloc(buf->Size());

  FILE *fp = fopen(fileName, "rb");

  fread(tmp, buf->Size(), 1, fp);

  buf->Set(tmp);

  fclose(fp);
}

void ConvertToFp16(gpu::Memory *input, gpu::Memory *output, int C, int H,
                   int W) {
  gpu::Device *dev_ = input->Dev();

  cl_int err = 0;
  cl_kernel kernel_convert_to_fp16 =
      clCreateKernel(dev_->ClDnnHandle()->program, "convert_to_fp16", &err);

  cl_mem inputMem = input->Ptr();
  cl_mem outputMem = output->Ptr();

  int outputChannel_ = C;
  int outputHeight_ = H;
  int outputWidth_ = W;

  err = clSetKernelArg(kernel_convert_to_fp16, 0, sizeof(cl_mem), &inputMem);
  err = clSetKernelArg(kernel_convert_to_fp16, 1, sizeof(cl_mem), &outputMem);
  err = clSetKernelArg(kernel_convert_to_fp16, 2, sizeof(cl_int),
                       &outputChannel_);
  err =
      clSetKernelArg(kernel_convert_to_fp16, 3, sizeof(cl_int), &outputHeight_);
  err =
      clSetKernelArg(kernel_convert_to_fp16, 4, sizeof(cl_int), &outputWidth_);

  size_t gws[3] = {(size_t)outputWidth_, (size_t)outputHeight_,
                   (size_t)outputChannel_};
  size_t lws[3] = {16, 4, 4};

  gws[0] = (gws[0] + lws[0] - 1) / lws[0] * lws[0];
  gws[1] = (gws[1] + lws[1] - 1) / lws[1] * lws[1];
  gws[2] = (gws[2] + lws[2] - 1) / lws[2] * lws[2];

  CheckCl(clEnqueueNDRangeKernel(dev_->ClDnnHandle()->queue,
                                 kernel_convert_to_fp16, 3, NULL, gws, lws, 0,
                                 NULL, NULL));
  output->SetUsedSize(outputChannel_ * outputHeight_ * outputWidth_ *
                      sizeof(DATA_TYPE));
}

void ConvertFromFp16(gpu::Memory *input, gpu::Memory *output, int C, int H,
                     int W) {
  gpu::Device *dev_ = input->Dev();

  cl_int err = 0;
  cl_kernel kernel_convert_from_fp16 =
      clCreateKernel(dev_->ClDnnHandle()->program, "convert_from_fp16", &err);

  cl_mem inputMem = input->Ptr();
  cl_mem outputMem = output->Ptr();

  int outputChannel_ = C;
  int outputHeight_ = H;
  int outputWidth_ = W;

  err = clSetKernelArg(kernel_convert_from_fp16, 0, sizeof(cl_mem), &inputMem);
  err = clSetKernelArg(kernel_convert_from_fp16, 1, sizeof(cl_mem), &outputMem);
  err = clSetKernelArg(kernel_convert_from_fp16, 2, sizeof(cl_int),
                       &outputChannel_);
  err = clSetKernelArg(kernel_convert_from_fp16, 3, sizeof(cl_int),
                       &outputHeight_);
  err = clSetKernelArg(kernel_convert_from_fp16, 4, sizeof(cl_int),
                       &outputWidth_);

  size_t gws[3] = {(size_t)outputWidth_, (size_t)outputHeight_,
                   (size_t)outputChannel_};
  size_t lws[3] = {16, 4, 4};

  gws[0] = (gws[0] + lws[0] - 1) / lws[0] * lws[0];
  gws[1] = (gws[1] + lws[1] - 1) / lws[1] * lws[1];
  gws[2] = (gws[2] + lws[2] - 1) / lws[2] * lws[2];

  CheckCl(clEnqueueNDRangeKernel(dev_->ClDnnHandle()->queue,
                                 kernel_convert_from_fp16, 3, NULL, gws, lws, 0,
                                 NULL, NULL));
  output->SetUsedSize(outputChannel_ * outputHeight_ * outputWidth_ *
                      sizeof(float));
}

static double get_time() {
  struct timeval tv;
  gettimeofday(&tv, 0);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

void timer_start(int i) { start_time[i] = get_time(); }

double timer_stop(int i) { return get_time() - start_time[i]; }

float *load(string filepath, size_t &num_elements_) {
  // file open
  std::ifstream fin(filepath, std::ios::binary);
  if (!fin) {
    cout << "Error, Couldn't find the file"
         << "\n";
    return 0;
  }

  fin.seekg(0, std::ios::end);
  const size_t num_elements = fin.tellg() / sizeof(float);
  fin.seekg(0, std::ios::beg);

  float *buffer = (float *)malloc(num_elements * sizeof(float));
  fin.read(reinterpret_cast<char *>(buffer), num_elements * sizeof(float));

  num_elements_ = num_elements;

  return buffer;
}

void write(string filepath, gpu::Memory *m) {
  int size = m->Size();
  char *buf = (char *)malloc(size);
  m->Get((void *)buf);

  std::ofstream fout(filepath, std::ios::binary);
  fout.write(buf, size);
  delete buf;
}

bool compare(gpu::Memory *input, gpu::Memory *target) {
  int size1 = input->Size();
  int size2 = target->Size();

  if (size1 != size2) {
    cout << "size incorrect" << endl;
    exit(-1);
  }

  DATA_TYPE *ptr1 = (DATA_TYPE *)malloc(size1);
  DATA_TYPE *ptr2 = (DATA_TYPE *)malloc(size1);

  input->Get((void *)ptr1);
  target->Get((void *)ptr2);

  int N = size1 / sizeof(DATA_TYPE);
  double total = 0, diff = 0, max = -1000000, min = 1000000;
  double mse = 0;
  printf("element:%d\n", N);
  int H, W, C;
  // C = 128;
  // H = 360;
  // W = 496;

  int c, h, w;

  for (int i = 0; i < N; i++) {
    {
      diff = abs(abs((float)(ptr1[i])) - abs(float(ptr2[i])));

      if (diff > 0.01) {
        c = i / (H * W);
        int hw = i % (H * W);
        h = hw / W;
        w = hw % W;
      }

      mse += diff * diff;

      total += diff;
      if (max < diff) max = diff;
      if (min > diff) min = diff;
    }
  }

  cout << "N : " << N << endl;
  mse /= N;
  cout << "mse : " << mse << endl;

  float psnr = 10 * log10f(1 / (float)mse);
  cout << "psnr:" << psnr << endl;

  cout << "total diff : " << total << endl;
  cout << "max diff : " << max << endl;
  cout << "min diff : " << min << endl;
  cout << "diff per pixel : " << total / N << endl;

  if (total / N < EPSILON)
    return true;
  else
    return false;
}

void SavePng(void *array, size_t size, int idx, int H, int W) {
  float *A = (float *)array;
  float *A_ = (float *)malloc(size);

  for (int i = 0; i < H * W; i++) {
    A_[i * 3 + 0] = A[i];
    A_[i * 3 + 1] = A[i + H * W];
    A_[i * 3 + 2] = A[i + 2 * (H * W)];
  }

  int numElem = size / sizeof(float);

  unsigned char *output =
      (unsigned char *)malloc(numElem * sizeof(unsigned char));
  for (int i = 0; i < numElem; i++) {
    output[i] = (unsigned char)(A_[i] * 255);
  }

  // save png
  FILE *fp = NULL;

  string path = "output_" + to_string(idx) + ".png";

  fp = fopen(path.c_str(), "wb");
  if (fp == NULL) {
    cout << "png file write error" << endl;
    exit(-1);
  }
  svpng(fp, W, H, output, 0);

  delete A_;
  delete output;
}

void checkError(string path, gpu::Memory *m) {
  float *buf;
  size_t num_element;
  buf = load(string("veri/") + path, num_element);
  cout << "--------------------------------" << endl;
  cout << "file path : " << path << endl;
  cout << "num elem : " << num_element << endl;
  gpu::Memory *t = new gpu::Memory(m->Dev(), num_element * sizeof(float));
  t->Set(buf);
  Print("pytorch output", t);

  // pass
  if (compare(t, m))
    cout << "validation pass" << endl;
  else
    cout << "validation fail" << endl;
  cout << "--------------------------------" << endl;

  delete t;
}
