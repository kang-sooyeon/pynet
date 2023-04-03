#include "memory.h"

#include <cassert>
#include <cstring>
#include <iostream>

namespace cpu {
size_t Memory::allocSizeAll_ = 0;

size_t Memory::AllocSizeAll() { return allocSizeAll_; }

void Memory::SetAllocSizeAll(size_t v) { allocSizeAll_ = v; }

Memory::Memory(Device *dev, size_t size)
    : allocSize_(size), usedSize_(size), dev_(dev) {
  ptr_ = malloc(allocSize_);
  assert(ptr_ != NULL);
  allocSizeAll_ += allocSize_;
}

void Memory::SetUsedSize(size_t usedSize) { usedSize_ = usedSize; }

Memory::~Memory() { free(ptr_); }

void *Memory::Ptr() { return ptr_; }

size_t Memory::Size() { return usedSize_; }

void Memory::Set(void *src) { memcpy(ptr_, src, usedSize_); }

void Memory::Clear() { memset(ptr_, 0, allocSize_); }

void Memory::Get(void *dst) { memcpy(dst, ptr_, usedSize_); }

cpu::Device *Memory::Dev() { return dev_; }

Memory &Memory::operator=(Memory &m) {
  float *src = reinterpret_cast<float *>(m.Ptr());
  float *dst = reinterpret_cast<float *>(Ptr());
  memcpy(dst, src, m.Size());
  SetUsedSize(m.Size());
}

void Memory::Print() {
  float *O = reinterpret_cast<float *>(Ptr());

  int flag = 2;

  switch (flag) {
    case 1:
      std::cout << std::endl;
      for (size_t i = 0; i < Size() / 4; i++) {
        std::cout << O[i] << ", ";
      }
      std::cout << std::endl;
      break;

    case 2:
      for (size_t i = 0; i < 25; i++) {
        std::cout << O[i] << ", ";
      }
      std::cout << std::endl;
      break;
  }
}

size_t Memory::Sumt() {
  size_t *O = reinterpret_cast<size_t *>(Ptr());
  size_t sum = 0;
  for (size_t i = 0; i < usedSize_ / sizeof(size_t); i++) {
    sum += O[i];
  }
  return sum;
}

double Memory::Sumf() {
  float *O = reinterpret_cast<float *>(Ptr());
  double sum = 0;
  for (size_t i = 0; i < usedSize_ / sizeof(float); i++) {
    sum += O[i];
  }
  return sum;
}

}  // namespace cpu
