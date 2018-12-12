#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>
#include <map>
#include "./common.hpp"
#include "./thread_local.hpp"

namespace caffe {
// Base class
struct PrvMemDescr {
  virtual ~PrvMemDescr() {}
  virtual void convert_from_prv(void* cpu_ptr) = 0;
  virtual void convert_to_prv(void* cpu_ptr) = 0;
  virtual void convert_from_other(shared_ptr<PrvMemDescr> other) = 0;
  virtual bool on_to_cpu() { return false; }
  virtual void* prv_ptr() = 0;
  // returns true for matching layouts
  virtual bool layout_compare(shared_ptr<PrvMemDescr> other) = 0;
  virtual size_t prv_count() = 0;
  virtual size_t prv_size() = 0;  // TODO: do we need both count() and size()?
  // This might help using prv_ptr_ by different accelerators/engines
  enum PrvDescrType {
    PRV_DESCR_MKL2017,
    PRV_DESCR_MKLDNN
  };
  virtual PrvDescrType get_descr_type() = 0;
};

class SyncedMemory {
 public:
  explicit SyncedMemory(size_t size)
      : cpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
       own_cpu_data_(false), own_prv_data_(false){}
  ~SyncedMemory();
  const void* cpu_data();
  void* mutable_cpu_data();
  void set_cpu_data(void* data);
  size_t size() { return size_; }
  shared_ptr<PrvMemDescr> prv_descriptor_;
  const void* prv_data();
  void* mutable_prv_data();
  void set_prv_descriptor(shared_ptr<PrvMemDescr> descriptor, bool same_data);
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, SYNCED,
                    HEAD_AT_PRV, SYNCED_PRV};
  SyncedHead head() { return head_; }
 private:
  void to_cpu();
  size_t size_;
  SyncedHead head_;
  void* cpu_ptr_;
  bool own_cpu_data_;
  bool own_prv_data_;
  std::mutex mtx;

  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};  // class SyncedMemory

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
