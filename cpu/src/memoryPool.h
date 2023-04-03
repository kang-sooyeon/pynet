
#include <vactor>

namespace cpu {

class MemoryPool {
 public:
  MemoryPool(Device* dev);
  ~MemoryPool();

  cpu::Memory* Malloc(size_t size);
  void Free(cpu::Memory* m);

 private:
  Device* dev_;
  BlockPool 1600M;
  BlockPool 1200M;
  BlockPool 800M;
  BlockPool 400M;
  BlockPool 200M;
  BlockPool 100M;
  BlockPool 50M;
  BlockPool 10M;
  BlockPool 1M;
}

class BlockPool {
  cpu::Memory* GetBlock() {
    // find allocated false block
    // if none, create block
  }

 private:
  vector<cpu::Memory*> blockPool;
}

}  // namespace cpu
