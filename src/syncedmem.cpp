#include <sstream>
#include <iomanip>
#include <mutex>
#include "./common.hpp"
#include "./syncedmem.hpp"
#include "./util/math_functions.hpp"
#include "common.hpp"

namespace caffe {


inline void CaffeMallocHost(void** ptr, size_t size) {
#ifdef USE_MKL
    *ptr = mkl_malloc(size ? size : 1, 64);
#else
   *ptr = malloc(size);
#endif
  CHECK(*ptr) << "host allocation of size " << size << " failed";
}

static void CaffeFreeHost(void* ptr) {
#ifdef USE_MKL
    mkl_free(ptr);
#else
    free(ptr);
#endif
}

SyncedMemory::~SyncedMemory() {
  if (cpu_ptr_ && own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_);
  }

}

inline void SyncedMemory::to_cpu() {
  switch (head_) {
  case UNINITIALIZED:
    CaffeMallocHost(&cpu_ptr_, size_);
    caffe_memset(size_, 0, cpu_ptr_);
    head_ = HEAD_AT_CPU;
    own_cpu_data_ = true;
    break;
  case HEAD_AT_PRV:
    if (cpu_ptr_ == NULL) {
      CaffeMallocHost(&cpu_ptr_, size_);
      own_cpu_data_ = true;
    }
    CHECK(prv_descriptor_.get());
    prv_descriptor_->convert_from_prv(cpu_ptr_);
    prv_descriptor_->on_to_cpu();
    head_ = SYNCED_PRV;
    break;
  case HEAD_AT_CPU:
  case SYNCED_PRV:
  case SYNCED:
    break;
  }
}

const void* SyncedMemory::cpu_data() {
  std::lock_guard<std::mutex> lock(mtx);
  to_cpu();
  return (const void*)cpu_ptr_;
}

void* SyncedMemory::mutable_cpu_data() {
  std::lock_guard<std::mutex> lock(mtx);
  to_cpu();
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
}
void SyncedMemory::set_cpu_data(void* data) {
  std::lock_guard<std::mutex> lock(mtx);
  CHECK(data);
  if (own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_);
  }
  cpu_ptr_ = data;
  head_ = HEAD_AT_CPU;
  own_cpu_data_ = false;
}

inline std::string MemSize(double size) {
  std::stringstream os;
  if (size < 1024.) {
    os << static_cast<int>(size) << " B";
  }
  else {
    size /= 1024.;
    os << std::setprecision(3);
    if (size < 1024.) {
      os << size << " K";
    }
    else {
      size /= 1024.;
      os << size << " M";
    }
  }
  return os.str();
}

inline bool ShouldBorrowMem(size_t has, size_t wants) {
  const int ratio = 2;
  return has / 2 <= wants;
}

//mkdnn
void SyncedMemory::set_prv_descriptor(shared_ptr<PrvMemDescr> descriptor,
        bool same_data) {
  std::lock_guard<std::mutex> lock(mtx);
  // If it wasn't synced before, it won't be now.
  if (descriptor == NULL) {
    if (head_ != UNINITIALIZED)
      head_ = HEAD_AT_CPU;
  } else {
    if ((head_ != HEAD_AT_PRV) && same_data)
      head_ = SYNCED_PRV;
    else
      head_ = HEAD_AT_PRV;
  }
  prv_descriptor_ = descriptor;
}

const void* SyncedMemory::prv_data() {
  if ((head_ != HEAD_AT_PRV) &&
     (head_ != SYNCED_PRV)) {
    return NULL;
  }

  CHECK(prv_descriptor_.get());
  return (const void* ) prv_descriptor_->prv_ptr();
}

void* SyncedMemory::mutable_prv_data() {
  std::lock_guard<std::mutex> lock(mtx);
  CHECK(prv_descriptor_.get());
  if (head_ == HEAD_AT_CPU) {
    prv_descriptor_->convert_to_prv(cpu_ptr_);
  }
  head_ = HEAD_AT_PRV;
  return prv_descriptor_->prv_ptr();
}

}  // namespace caffe
