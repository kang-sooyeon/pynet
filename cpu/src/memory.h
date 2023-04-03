
#ifndef _CPU_MEMORY_H_
#define _CPU_MEMORY_H_

#include <cstdlib>

#include "device.h"
#include "memory.h"

namespace cpu {

class Memory {
 public:
  Memory(Device *dev, size_t size);
  ~Memory();
  void *Ptr();
  size_t Size();  // used size
  size_t AllocSizeAll();
  void Set(void *src);
  void Get(void *dst);
  void Clear();
  cpu::Device *Dev();
  void Print();
  double Sumf();
  size_t Sumt();
  void SetAllocSizeAll(size_t v);
  void SetUsedSize(size_t usedSize);

  Memory &operator=(Memory &m);

 private:
  static size_t allocSizeAll_;
  size_t allocSize_;
  size_t usedSize_;
  Device *dev_;
  void *ptr_;
};

}  // namespace cpu

#endif  // CPU_MEMORY_H_
