#ifndef CAFFE_MKL_MEMORY_HPP_
#define CAFFE_MKL_MEMORY_HPP_

#ifdef USE_MKLDNN
#include "../../syncedmem.hpp"
#include "mkldnn.hpp"
#include "mkldnn_base.hpp"
#include <memory>
#include <caffe/blob.hpp>

namespace caffe {
using namespace mkldnn;
class MKLDNNLayer;
class MKLDNNMemoryDescriptorBase : public PrvMemDescr
        , public std::enable_shared_from_this<MKLDNNMemoryDescriptorBase >
{
public:
    MKLDNNMemoryDescriptorBase(std::shared_ptr<memory::primitive_desc> usr_memory_pd
                                , shared_ptr<memory::primitive_desc> prv_memory_pd
                                , Blob* blob, MKLDNNLayer* mkldnn_layer
                                , std::vector<float>scale=std::vector<float>(1,1.)
                                , int mask=0
                                , bool is_sum=false
                                , bool is_wino=false);


    ~MKLDNNMemoryDescriptorBase() {}
    // ---- PrvMemDescr virtual functions -----
    virtual void convert_from_other(shared_ptr<PrvMemDescr> other);
    virtual bool layout_compare(shared_ptr<PrvMemDescr> other);
    virtual PrvDescrType get_descr_type() {return PRV_DESCR_MKLDNN;}

    // TODO: assuming size/sizeof = count may be not correct
    virtual size_t prv_count() { return prv_size()/sizeof(real_t); }

    virtual size_t prv_size() { return _prv_memory_pd->get_size(); }
    // ---------------------------------------
    std::shared_ptr<MKLDNNMemoryDescriptorBase > get_shared_ptr() {
        return this->shared_from_this();
    }
    shared_ptr<memory::primitive_desc>  prv_memory_pd() const {
        return _prv_memory_pd;
    }
    shared_ptr<memory::primitive_desc>  usr_memory_pd() const {
        return _usr_memory_pd;
    }
    inline bool conversion_needed() const { return (_reorder_usr2prv_pd != NULL || _reorder_extprv2prv_pd != NULL); }
    virtual void* prv_ptr() { return _internal_ptr;  }

    shared_ptr<memory>  get_prv_memory()
    {
        if (_prv_memory == NULL) allocate();
        return _prv_memory;
    }
    real_t * get_prv_ptr() {
        if (_prv_memory == NULL) allocate();
        return _internal_ptr;
    }

    shared_ptr<primitive>  reorder_usr2prv() { return _reorder_usr2prv.aprimitive; }
    shared_ptr<primitive>  reorder_prv2usr() { return _reorder_prv2usr.aprimitive; }
    shared_ptr<primitive>  reorder_extprv2prv() { return _reorder_extprv2prv.aprimitive; }

    float get_scale(int i) { return _scale[i]; }
    std::vector<float> get_scale() { return _scale; }
    void set_scale(std::vector<float> scale) { _scale.assign(scale.begin(),scale.end());}

    void set_sum(bool is_sum) { _is_sum = is_sum; }
    bool get_sum() { return _is_sum; }

    void set_mkldnn_layer(MKLDNNLayer* layer) { _mkldnn_layer = layer;  }
    MKLDNNLayer*  mkldnn_layer() const { return _mkldnn_layer;  }

    std::string name;  // for debugging purposes
protected:
    void check_usr_with_prv_descriptors();
    void set_prv_memory(shared_ptr<memory> memory)
    {
        _prv_memory = memory;
        _internal_ptr = (real_t *)(_prv_memory->get_data_handle());
    }

    void allocate() {
        if (_prv_memory == NULL) {
            _prv_memory = shared_ptr<memory>(new memory(*_prv_memory_pd));
          _internal_ptr = (real_t *)(_prv_memory->get_data_handle());
          // TODO: may need initialize memory by 0
        }
    }
    void set_prv_memory_pd(shared_ptr<memory::primitive_desc> memory_pd, std::vector<float> scale, int mask, bool is_wino)  {
        _prv_memory_pd = memory_pd;
        if (_prv_memory_pd && _usr_memory_pd) {
            check_usr_with_prv_descriptors();
            std::vector<float>scale_ext = std::vector<float>(1,1.);
            this->create_reorder_descriptors(scale, mask, scale_ext, false, is_wino);
        }
    }

    void set_extprv_memory_pd(shared_ptr<memory::primitive_desc> memory_pd, std::vector<float> scale, std::vector<float> scale_ext, bool is_sum)  {
        _extprv_memory_pd = memory_pd;
        if (_prv_memory_pd && _usr_memory_pd) {
            check_usr_with_prv_descriptors();
            this->create_reorder_descriptors(scale, 0, scale_ext, is_sum);
        }
    }

    void set_usr_memory_pd(shared_ptr<memory::primitive_desc> memory_pd, std::vector<float> scale) {
        _usr_memory_pd = memory_pd;
    }

    void create_reorder_descriptors(std::vector<float> scale, int mask=0, std::vector<float>scale_ext=std::vector<float>(1,1.), bool is_sum=false, bool is_wino=false);

    shared_ptr<memory::primitive_desc> _usr_memory_pd;
    shared_ptr<memory::primitive_desc> _prv_memory_pd;
    shared_ptr<memory::primitive_desc> _extprv_memory_pd;
    shared_ptr<reorder::primitive_desc> _reorder_usr2prv_pd;
    shared_ptr<reorder::primitive_desc> _reorder_prv2usr_pd;
    shared_ptr<reorder::primitive_desc> _reorder_extprv2prv_pd;
    MKLDNNPrimitive _reorder_usr2prv;
    MKLDNNPrimitive _reorder_prv2usr;
    MKLDNNPrimitive _reorder_extprv2prv;
    shared_ptr<memory> _prv_memory;
    real_t * _internal_ptr;
    shared_ptr<memory> _usr_memory;
    void* _cpu_ptr;

    MKLDNNLayer* _mkldnn_layer;
    Blob* _blob;
    std::vector<float> _scale = std::vector<float>(1,1.);
    bool _is_sum = false;
};


class MKLDNNMemoryDescriptor : public MKLDNNMemoryDescriptorBase {
public:
    MKLDNNMemoryDescriptor(shared_ptr<memory::primitive_desc> usr_memory_pd
                        , shared_ptr<memory::primitive_desc> prv_memory_pd
                        , Blob* blob, MKLDNNLayer* mkldnn_layer
                        , std::vector<float> scale=std::vector<float>(1,1.)
                        , int mask=0
                        , bool is_sum=false
                        , bool is_wino=false);

    virtual void convert_from_prv(void* cpu_ptr);
    virtual void convert_to_prv(void* cpu_ptr);
    virtual void convert_from_extprv(shared_ptr<primitive> aprimitive);
    virtual bool on_to_cpu();

    virtual void create_reorder_from_prv(void* cpu_ptr);
    virtual void create_reorder_to_prv(void* cpu_ptr);
    virtual void create_reorder_from_extprv(shared_ptr<primitive> aprimitive);

    // The last get_blob_data_ptr() argument is a hack for reusing
    // in backward a conversion done already in the forward direction.
    shared_ptr<primitive> get_blob_prv_primitive(Blob * blob, bool set_prv_ptr, bool convert = true,
            MKLDNNMemoryDescriptor* converted_in_fwd = NULL);

    void sync_before_read();
    void sync_before_write(bool inplace = false);

    shared_ptr<primitive> create_input(Blob * blob, bool set_prv_ptr);
    shared_ptr<memory> create_output_memory(Blob * blob, bool inplace = false);
    shared_ptr<primitive> create_input(bool set_prv_ptr);
    shared_ptr<memory> create_output_memory(bool inplace = false);
    real_t* get_memory_ptr(long offset = 0);
    shared_ptr<memory::desc> get_memory_desc();
    size_t get_memory_count();
    void set_mkldnn_primitive(MKLDNNPrimitive& mprimitive) { CHECK(mprimitive.aprimitive); _mkldnn_primitive = mprimitive;  }
    MKLDNNPrimitive&  mkldnn_primitive() { return _mkldnn_primitive; }
    shared_ptr<primitive> aprimitive() const { return _mkldnn_primitive.aprimitive; }
private:
    MKLDNNPrimitive _mkldnn_primitive;
};

class MKLDNNData : public MKLDNNMemoryDescriptor
{
public:
    MKLDNNData(shared_ptr<memory::primitive_desc> usr_memory_pd
                , shared_ptr<memory::primitive_desc> prv_memory_pd
                , Blob* blob, MKLDNNLayer* mkldnn_layer
                , std::vector<float> scale=std::vector<float>(1,1.)
                , int mask=0
                , bool is_sum=false
                , bool is_wino=false)
        : MKLDNNMemoryDescriptor(usr_memory_pd, prv_memory_pd, blob, mkldnn_layer, scale, mask, is_sum, is_wino) {}
};

shared_ptr<MKLDNNMemoryDescriptor > get_mkldnn_prv_descriptor(Blob* blob);
}
#endif // use mkldnn
#endif  // #ifndef CAFFE_MKL_MEMORY_HPP_