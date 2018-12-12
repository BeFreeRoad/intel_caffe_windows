#include <vector>

#include "./split_layer.hpp"
#include "../util/math_functions.hpp"
#ifdef USE_MKLDNN
#include "./intel/mkldnn_layers.hpp"
#endif
namespace caffe {

void SplitLayer::Reshape(const vector<Blob*>& bottom,
                         const vector<Blob*>& top) {
  count_ = bottom[0]->count();
  for (int i = 0; i < top.size(); ++i) {
    // Do not allow in-place computation in the SplitLayer.  Instead, share data
    // by reference in the forward pass, and keep separate diff allocations in
    // the backward pass.  (Technically, it should be possible to share the diff
    // blob of the first split output with the input, but this seems to cause
    // some strange effects in practice...)
    CHECK_NE(top[i], bottom[0]) << this->type() << " Layer does not "
        "allow in-place computation.";
    top[i]->ReshapeLike(*bottom[0]);
    CHECK_EQ(count_, top[i]->count());
  }
}

void SplitLayer::Forward_cpu(const vector<Blob*>& bottom,
                             const vector<Blob*>& top) {
  for (int i = 0; i < top.size(); ++i) {
    caffe_copy(bottom[0]->count(), bottom[0]->cpu_data(), top[i]->mutable_cpu_data());
  }
}

void SplitLayer::Forward_gpu(const vector<Blob*>& bottom,
                             const vector<Blob*>& top) {
}

// Creator
static shared_ptr<Layer> CreateLayer(const LayerParameter &param) {
#ifdef USE_MKLDNN
  return shared_ptr<Layer>(new MKLDNNSplitLayer(param));
#endif
  return shared_ptr<Layer>(new SplitLayer(param));
}

REGISTER_LAYER_CREATOR(Split, CreateLayer);
}  // namespace caffe
