#ifndef CAFFE_MKLDNN_BASE_HPP_
#define CAFFE_MKLDNN_BASE_HPP_

#ifdef USE_MKLDNN
#include "../../syncedmem.hpp"
#include "mkldnn.hpp"
#include <memory>

namespace caffe {
using namespace mkldnn;

// =====  CpuEngine =======================================
// cpu_engine singleton
class CpuEngine
{
public:
    static CpuEngine & Instance()
    {
        // I's thread-safe in C++11.
        static CpuEngine myInstance;
        return myInstance;
    }
    CpuEngine(CpuEngine const&) = delete;             // Copy construct
    CpuEngine(CpuEngine&&) = delete;                  // Move construct
    CpuEngine& operator=(CpuEngine const&) = delete;  // Copy assign
    CpuEngine& operator=(CpuEngine &&) = delete;      // Move assign

    engine & get_engine() { return _cpu_engine; }
protected:
    CpuEngine() : _cpu_engine(engine::cpu, 0) {}
//    CpuEngine() : _cpu_engine(engine::cpu_lazy, 0) {}
    ~CpuEngine() {}
private:
    engine _cpu_engine;
};

// =====  MKLDNNStream =======================================
class MKLDNNStream {
public:
    explicit MKLDNNStream():_ready(false) { prepare(); }
    virtual ~MKLDNNStream() {}
    MKLDNNStream  &submit(std::vector<primitive> primitives) { _stream->submit(primitives); return *this; }
    bool wait(bool block = true) {
        VLOG(1) << typeid(*this).name()<< " : " << __FUNCTION__ << " : wait stream ";
        _ready = false;
        bool res = _stream->wait(block);
        VLOG(1) << typeid(*this).name()<< " : " << __FUNCTION__ << " : end of stream waiting ";
        return res;
    }
    bool ready() { return _ready; }
    void prepare() {
        if(_ready == false) {
            // stream just created or already executed
            // !! TODO: change below if stream will have method to reset its state
            VLOG(1) << typeid(*this).name()<< " : " << __FUNCTION__ << " : create new stream";
//            _stream.reset(new stream(stream::kind::any));
            _stream.reset(new stream(stream::kind::eager));
            // TODO: Enable when Unit tests work for this one
            //_stream.reset(new stream(stream::kind::lazy));
        }
        _ready = true;
    }
protected:
private:
    bool _ready;
    shared_ptr<stream> _stream;
};


// =====  StreamHolder =======================================
// singleton
class StreamHolder
{
public:
    static StreamHolder & Instance()
    {
        // I's thread-safe in C++11.
        static StreamHolder myInstance;
        return myInstance;
    }
    StreamHolder(StreamHolder const&) = delete;             // Copy construct
    StreamHolder(StreamHolder&&) = delete;                  // Move construct
    StreamHolder& operator=(StreamHolder const&) = delete;  // Copy assign
    StreamHolder& operator=(StreamHolder &&) = delete;      // Move assign

    shared_ptr<MKLDNNStream> get_stream();
    shared_ptr<MKLDNNStream> current_stream() { return _current_stream; }
    void prepare_mkldnn_stream(shared_ptr<MKLDNNStream> mkldnn_stream) {
        _current_stream = mkldnn_stream;
        _current_stream->prepare();
    }
protected:
    StreamHolder() : _current_stream() {}
    ~StreamHolder() {}
private:
    shared_ptr<MKLDNNStream> _current_stream;
};

class MKLDNNPrimitive {
public:
    explicit MKLDNNPrimitive() : aprimitive(), mkldnn_stream() {}

    //API for initializing with shared_ptr<primitive>
    MKLDNNPrimitive(shared_ptr<primitive> aprimitive_input) { this->aprimitive = aprimitive_input; }

    virtual ~MKLDNNPrimitive() {}

    void reset(primitive *pprimitive) { this->aprimitive.reset(pprimitive); }

    shared_ptr<primitive> aprimitive;
    shared_ptr<MKLDNNStream> mkldnn_stream;

    shared_ptr<MKLDNNStream> get_mkldnn_stream();

    shared_ptr<MKLDNNStream> submit();
};

}//namespace caffe

#endif // USE_MKLDNN
#endif  // #ifndef CAFFE_MKLDNN_BASE_HPP_
