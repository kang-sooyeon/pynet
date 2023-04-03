
namespace cpu {

MemoryPool::MemoryPool(Device* dev) : dev_(dev) {}

MemoryPool::~MemoryPool() {
  // free all pool
}

cpu::Memory* MemoryPool::Malloc(size_t size) {
  size = Round(size);
  BlockPool bp = GetPool(size);

  return bp.Get();
}

cpu::Memory* MemoryPool::GetPool(size_t size) {
  switch (size) {
    case 1600:
      return 1600M;
    case 1200:
      return 1200M;
  }
}

void MemoryPool::Free(cpu::Memory* m) {
  int id = m->GetMemID();
  int allocSize = m->GetAllocSize();

  BlockPool blockPool = GetPool(allocSize);
  int idx = find(pool, id);
  pool[idx]->SetAllocated(false);
}

void clean(int level) {
  if (level == 1) {
    // clean 1200M
    // clean 1600M
  } else if (level == 2) {
  }
}
}  // namespace cpu
