#ifdef USE_MKLDNN
#include <algorithm>
#include <cfloat>
#include <vector>

#include <algorithm>
#include <vector>
#include "engine_parser.hpp"
#include "mkldnn_layers.hpp"

namespace caffe {

MKLDNNSplitLayer::~MKLDNNSplitLayer() { }

void MKLDNNSplitLayer::Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  size_t count = bottom[0]->count();
  for (int i = 0; i < top.size(); ++i) {
    // Do not allow in-place computation in the SplitLayer.  Instead, share data
    // by reference in the forward pass, and keep separate diff allocations in
    // the backward pass.  (Technically, it should be possible to share the diff
    // blob of the first split output with the input, but this seems to cause
    // some strange effects in practice...)
    CHECK_NE(top[i], bottom[0]) << this->type() << " Layer does not "
        "allow in-place computation.";
    top[i]->ReshapeLike(*bottom[0]);
    CHECK_EQ(count, top[i]->count());
  }
  size_t dim_src = bottom[0]->shape().size();
  this->reshape = false;
  if (this->sizes_src_.size() != dim_src || this->strides_src_.size() != dim_src) {
    this->sizes_src_.resize(dim_src);
    this->strides_src_.resize(dim_src);
    this->reshape = true;
  }
  for (size_t d = 0; d < dim_src; ++d) {
    if (this->sizes_src_[d] != bottom[0]->shape()[d]) {
      this->sizes_src_[d] = bottom[0]->shape()[d];
      this->reshape = true;
    }
    size_t stride = (d == 0) ? 1 : this->strides_src_[d-1]*this->sizes_src_[d-1];
    if (this->strides_src_[d] != stride) {
      this->strides_src_[d] = stride;
      this->reshape = true;
    }
  }

  // TODO: Add checking to reinitialize Backward, to be
  // done when Reshape is to be supported by MKLDNN layers
}


void MKLDNNSplitLayer::LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
}

void MKLDNNSplitLayer::Forward_cpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  for (int i = 0; i < top.size(); ++i) {
    top[i]->ShareData(*bottom[0]);
  }
}

} // namespace caffe

#endif
