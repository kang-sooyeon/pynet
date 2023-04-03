
#include <sys/time.h>
#include <util.h>

#include <cstring>
#include <fstream>

#include "svpng.inc"

#define EPSILON 0.01
static double start_time[8];

void Print(char *layerName, cpu::Memory *m) {
  std::cout << "------------------------------------------------------------"
            << std::endl;
  std::cout << layerName << std::endl;
  m->Print();
  std::cout << "Sum : " << m->Sumf() << std::endl;
  std::cout << "Size : " << m->Size() << std::endl;
  std::cout << "------------------------------------------------------------"
            << std::endl;
}

static double get_time() {
  struct timeval tv;
  gettimeofday(&tv, 0);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

void timer_start(int i) { start_time[i] = get_time(); }

double timer_stop(int i) { return get_time() - start_time[i]; }

float *load(string filepath, size_t &size) {
  // file open
  std::ifstream fin(filepath, std::ios::binary);
  if (!fin) {
    cout << "Error, Couldn't find the file" << endl;
    exit(-1);
  }

  fin.seekg(0, std::ios::end);
  const size_t num_elements = fin.tellg() / sizeof(float);
  fin.seekg(0, std::ios::beg);

  float *buffer = (float *)malloc(num_elements * sizeof(float));
  fin.read(reinterpret_cast<char *>(buffer), num_elements * sizeof(float));

  size = num_elements;

  return buffer;
}

bool compare(cpu::Memory *input, cpu::Memory *target) {
  float *ptr1 = (float *)input->Ptr();
  float *ptr2 = (float *)target->Ptr();

  int size1 = input->Size();
  int size2 = target->Size();

  if (size1 != size2) {
    cout << "size incorrect" << endl;
    exit(-1);
  }

  int N = size1 / sizeof(float);
  float total = 0, diff, max = -1000000, min = 1000000;

  for (int i = 0; i < N; i++) {
    if (abs(ptr1[i] - ptr2[i])) {
      // cout << "i : " << i << ", ptr1 : " << ptr1[i] << ", ptr2 : " << ptr2[i]
      // << ", diff : " << diff << endl; cout << "ih : " << i/3968 << endl; cout
      // << "iw : " << i%3968 << endl;
      //
      total += diff;
      if (max < diff) max = diff;
      if (min > diff) min = diff;
    }
  }

  cout << "total diff : " << total << endl;
  cout << "max diff : " << max << endl;
  cout << "min diff : " << min << endl;
  cout << "diff per pixel : " << total / N << endl;

  if (total / N > EPSILON)
    return true;
  else
    return false;
}

void checkError(string path, cpu::Memory *m) {
  float *buf;
  size_t num_element;
  buf = load(path, num_element);
  cout << "num elem : " << num_element << endl;
  cpu::Memory *t = new cpu::Memory(m->Dev(), num_element * sizeof(float));
  t->Set(buf);

  // pass
  if (compare(t, m))
    cout << "validation pass" << endl;
  else
    cout << "validation fail" << endl;

  delete t;
}

void SavePng(float *array, size_t size, int idx, int H, int W) {
  int numElem = size / sizeof(float);

  // convert dim (c, h, w) --> (h, w, c)
  float *_output = (float *)malloc(size);

  for (int i = 0; i < H * W; i++) {
    _output[i * 3 + 0] = array[i];
    _output[i * 3 + 1] = array[i + H * W];
    _output[i * 3 + 2] = array[i + 2 * (H * W)];
  }

  // convert float to unsigned char
  unsigned char *output =
      (unsigned char *)malloc(numElem * sizeof(unsigned char));
  for (int i = 0; i < numElem; i++) {
    output[i] = (unsigned char)(_output[i] * 255);
  }

  // save png
  FILE *fp = NULL;
  //#ifdef USE_FLOAT_PRECISION
  string path = "output_32" + to_string(idx) + ".png";
  //#else
  // string path = "output_16" + to_string(idx) + ".png";
  //#endif
  fp = fopen(path.c_str(), "wb");
  if (fp == NULL) {
    cout << "png file write error" << endl;
    exit(-1);
  }
  svpng(fp, W, H, output, 0);

  delete _output;
  delete output;
}
