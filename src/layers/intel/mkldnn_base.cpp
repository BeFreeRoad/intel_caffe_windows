#ifdef USE_MKLDNN
#include "mkldnn_memory.hpp"

namespace caffe {
shared_ptr<MKLDNNStream> StreamHolder::get_stream()
{
    if (this->_current_stream == NULL || !this->_current_stream->ready()) {
        _current_stream.reset(new MKLDNNStream());
    }
    return _current_stream;
}

shared_ptr<MKLDNNStream>  MKLDNNPrimitive::get_mkldnn_stream() {
    if(mkldnn_stream == NULL)
        mkldnn_stream = StreamHolder::Instance().get_stream();
    else
        StreamHolder::Instance().prepare_mkldnn_stream(mkldnn_stream);
    return mkldnn_stream;

}
shared_ptr<MKLDNNStream> MKLDNNPrimitive::submit() {
    CHECK(this->aprimitive);
    this->get_mkldnn_stream()->submit({*(this->aprimitive)});
    return mkldnn_stream;
}

}
#endif  // #ifdef MKLDNN_SUPPORTED