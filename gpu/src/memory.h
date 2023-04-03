
#ifndef _GPU_MEMORY_H_
#define _GPU_MEMORY_H_

#include <device.h>

#include <cstdlib>
#include <vector>

namespace gpu {

class Memory {
 public:
  Memory(Device *dev);
  Memory(Device *dev, size_t size);
  ~Memory();
  void CreateBuffer(size_t size, bool is_alloc_host_ptr = true);
  void Release();
  cl_mem Ptr();
  size_t Size();  // used size
  size_t AllocSizeAll();
  void Set(void *src);
  void Get(void *dst);
  Device *Dev();
  void Print();
  void Print(int n);
  double Sumf();
  size_t Sumt();
  void SetAllocSizeAll(size_t v);
  void SetUsedSize(size_t usedSize);

  void addImage(cl_mem useImage);
  void releaseImages();

  cl_mem createImage(int inputImageSize);
  cl_mem createImage(cl_mem buf, size_t imageSize);

  cl_mem createImage2(int inputImageSize);
  cl_mem createImage2(cl_mem buf, int inputImageSize);
  cl_mem createImageFloat2(int inputImageSize);
  cl_mem createImageFloat2(cl_mem buf, int inputImageSize);

  cl_mem createImage2D(int inputImageSizeX, int inputImageSizeY);
  cl_mem createImage2D(cl_mem buf, int inputImageSizeX, int inputImageSizeY);

 private:
  static size_t allocSizeAll_;
  size_t allocSize_;
  size_t usedSize_;
  Device *dev_;
  cl_mem ptr_;

  std::vector<cl_mem> images_;
};

}  // namespace gpu

#endif  // GPU_MEMORY_H_
