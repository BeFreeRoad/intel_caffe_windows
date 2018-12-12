#ifdef USE_MKLDNN
#include "mkldnn_memory.hpp"

namespace caffe {
MKLDNNMemoryDescriptorBase::MKLDNNMemoryDescriptorBase(shared_ptr<memory::primitive_desc> usr_memory_pd
                                                            , shared_ptr<memory::primitive_desc> prv_memory_pd
                                                            , Blob* blob
                                                            , MKLDNNLayer* mkldnn_layer
                                                            , std::vector<float> scale
                                                            , int mask
                                                            , bool is_sum
                                                            , bool is_wino)
                                    : name("MKLDNNMemoryDescriptorBase")
                                    , _reorder_usr2prv_pd(), _reorder_prv2usr_pd(), _reorder_extprv2prv_pd()
                                    ,_prv_memory(), _internal_ptr(NULL), _usr_memory(), _cpu_ptr(NULL)
                                    , _mkldnn_layer(NULL)
{
    set_usr_memory_pd(usr_memory_pd, scale);
    set_prv_memory_pd(prv_memory_pd, scale, mask, is_wino);
    set_mkldnn_layer(mkldnn_layer);
    this->set_scale(scale);
    this->set_sum(is_sum);
    this->_blob = blob;
}
shared_ptr<MKLDNNMemoryDescriptor> get_mkldnn_prv_descriptor(Blob* blob)
{
    shared_ptr<PrvMemDescr> blob_prv_mem_descriptor = blob->get_prv_data_descriptor();

    CHECK_EQ(blob_prv_mem_descriptor->get_descr_type(), PrvMemDescr::PRV_DESCR_MKLDNN);

    shared_ptr<MKLDNNMemoryDescriptor > blob_prv_mkldnn_mem_descr =std::static_pointer_cast<MKLDNNMemoryDescriptor >(blob_prv_mem_descriptor);
    CHECK(blob_prv_mkldnn_mem_descr != NULL);
    return blob_prv_mkldnn_mem_descr;
}

MKLDNNMemoryDescriptor::MKLDNNMemoryDescriptor(shared_ptr<memory::primitive_desc> usr_memory_pd
                        , shared_ptr<memory::primitive_desc> prv_memory_pd
                        , Blob* blob, MKLDNNLayer* mkldnn_layer
                        , std::vector<float> scale
                        , int mask
                        , bool is_sum
                        , bool is_wino)
        : MKLDNNMemoryDescriptorBase(usr_memory_pd, prv_memory_pd, blob, mkldnn_layer, scale, mask, is_sum, is_wino)
{
    const real_t * prv_ptr = blob->prv_data();

    if (prv_ptr != NULL) {
        shared_ptr<MKLDNNMemoryDescriptor > blob_prv_mkldnn_mem_descr = get_mkldnn_prv_descriptor(blob);
        if (blob_prv_mkldnn_mem_descr->prv_memory_pd()->desc().data.format !=  this->prv_memory_pd()->desc().data.format || blob_prv_mkldnn_mem_descr->prv_memory_pd()->desc().data.data_type !=  this->prv_memory_pd()->desc().data.data_type || blob_prv_mkldnn_mem_descr->get_scale() != this->get_scale()) {
            this->set_extprv_memory_pd(blob_prv_mkldnn_mem_descr->prv_memory_pd(), scale, blob_prv_mkldnn_mem_descr->get_scale(), blob_prv_mkldnn_mem_descr->get_sum());
        }
    }
}

void MKLDNNMemoryDescriptor::convert_from_prv(void* cpu_ptr)
{
    CHECK(cpu_ptr);
    if(this->_reorder_prv2usr_pd == NULL)
        return;
    create_reorder_from_prv(cpu_ptr);
    VLOG(1) << "--- MKLDNNMemoryDescriptorBase::convert_from_prv --- " << this->name;
    this->_reorder_prv2usr.submit();
}


void MKLDNNMemoryDescriptor::convert_to_prv(void* cpu_ptr)
{
    create_reorder_to_prv(cpu_ptr);
//    VLOG(1) << "--- MKLDNNMemoryDescriptorBase::convert_to_prv --- " << this->name;
    this->_reorder_usr2prv.submit();
}

void MKLDNNMemoryDescriptor::convert_from_extprv(shared_ptr<primitive> aprimitive)
{
    CHECK(aprimitive);
    if(this->_reorder_extprv2prv_pd == NULL)
        return;
//    if (*this->_extprv_memory_pd == *this->_prv_memory_pd)
//    {
//#ifdef DEBUG
//        LOG(INFO) << "The format and data_type of _extprv_memory_pd and _prv_memory_pd is same, no need do conversion.";
//#endif
//        return;
//    }
    create_reorder_from_extprv(aprimitive);
//    VLOG(1) << "--- MKLDNNMemoryDescriptorBase::convert_from_extprv --- " << this->name;
    this->_reorder_extprv2prv.submit();
}

bool MKLDNNMemoryDescriptor::on_to_cpu()
{
    CHECK(this->mkldnn_layer());
    if (StreamHolder::Instance().current_stream() != NULL && StreamHolder::Instance().current_stream()->ready()) {
        VLOG(1) << "- MKLDNNMemoryDescriptorBase::" << __FUNCTION__ << ": stream.wait() - " << this->name;
        StreamHolder::Instance().current_stream()->wait();
    }
    return true;
}

void MKLDNNMemoryDescriptor::create_reorder_from_prv(void* cpu_ptr)
{
    CHECK(cpu_ptr);
    CHECK(this->_usr_memory_pd);
    CHECK(this->_prv_memory_pd);
    CHECK(this->_reorder_prv2usr_pd);
    if(this->_usr_memory == NULL || this->_cpu_ptr != cpu_ptr)
        this->_usr_memory.reset(new memory(*this->_usr_memory_pd, cpu_ptr));
    if(this->_reorder_prv2usr.aprimitive == NULL || this->_cpu_ptr != cpu_ptr) {
        CHECK(this->aprimitive());
        this->_reorder_prv2usr.aprimitive.reset(new reorder(*this->_reorder_prv2usr_pd, *this->aprimitive(), *this->_usr_memory));
    }
    this->_cpu_ptr = cpu_ptr;
}

void MKLDNNMemoryDescriptor::create_reorder_to_prv(void* cpu_ptr)
{
    CHECK(cpu_ptr);
    CHECK(this->_usr_memory_pd);
    CHECK(this->_prv_memory_pd);
    CHECK(this->_reorder_usr2prv_pd);

    if(this->_usr_memory == NULL || this->_cpu_ptr != cpu_ptr)
        this->_usr_memory.reset(new memory(*this->_usr_memory_pd, cpu_ptr));
    if(this->_reorder_usr2prv.aprimitive == NULL || this->_cpu_ptr != cpu_ptr)
        this->_reorder_usr2prv.reset(new reorder(*this->_reorder_usr2prv_pd, *this->_usr_memory, *this->get_prv_memory()));

    this->_cpu_ptr = cpu_ptr;
}

void MKLDNNMemoryDescriptor::create_reorder_from_extprv(shared_ptr<primitive> aprimitive)
{
    CHECK(aprimitive);
    CHECK(this->_extprv_memory_pd);
    CHECK(this->_prv_memory_pd);
    CHECK(this->_reorder_extprv2prv_pd);
    if(this->_reorder_extprv2prv.aprimitive == NULL)
        this->_reorder_extprv2prv.reset(new reorder(*this->_reorder_extprv2prv_pd, *aprimitive, *this->get_prv_memory()));
}


shared_ptr<primitive> MKLDNNMemoryDescriptor::get_blob_prv_primitive(Blob* blob
                                            ,bool set_prv_ptr, bool convert
                                            ,MKLDNNMemoryDescriptor* converted_in_fwd)
{
    if (!this->conversion_needed()) {
        return shared_ptr<primitive>(); // TODO: may be CHECK ?
    }

    // Conversion is needed
    const real_t* prv_ptr = blob->prv_data();
    if (prv_ptr == NULL) {
        if (converted_in_fwd) {
            // TODO: use previously done conversion on forward - needed for training
            NOT_IMPLEMENTED;
        }
        if(convert) {
            this->convert_to_prv(const_cast<real_t*> (blob->cpu_data()));
        }
        else {
            this->create_reorder_to_prv(const_cast<real_t*> (blob->cpu_data()));
        }
        if (set_prv_ptr) {
            blob->set_prv_data_descriptor(this->get_shared_ptr(), false);
                // below line designated to set correspondent SyncedMemory->_head to HEAD_AT_CPU
                // TODO: need to optimize
            blob->set_prv_data_descriptor(NULL);
        }
        return this->reorder_usr2prv();
    } else {
        shared_ptr<MKLDNNMemoryDescriptor > blob_prv_mkldnn_mem_descr = get_mkldnn_prv_descriptor(blob);
        if ((*blob_prv_mkldnn_mem_descr->prv_memory_pd() !=  *this->prv_memory_pd() || blob_prv_mkldnn_mem_descr->get_scale() != this->get_scale()) && this->_reorder_extprv2prv_pd != NULL) {
            // prv in blob and in this descrptor may have different layouts
            if(convert) {
                LOG(INFO) << "BAD CONVERT";
                this->convert_from_extprv(blob_prv_mkldnn_mem_descr->aprimitive());
            }
            else {
                this->create_reorder_from_extprv(blob_prv_mkldnn_mem_descr->aprimitive());
            }
            return this->reorder_extprv2prv();
        } else if (blob_prv_mkldnn_mem_descr.get() != this) {
//            VLOG(1) << "layout OK " << blob_prv_mkldnn_mem_descr->name << " == " << this->name;
        }
        return blob_prv_mkldnn_mem_descr->aprimitive();
    }
    NOT_IMPLEMENTED;
    return shared_ptr<mkldnn::primitive>();
}

void MKLDNNMemoryDescriptor::sync_before_read()
{
    // TODO: need to optimize code
    if (!this->conversion_needed()) {
        return;
    }

    // Conversion is needed
    const real_t* prv_ptr = this->_blob->prv_data();
    if (prv_ptr == NULL) {
        this->convert_to_prv(const_cast<real_t*> (this->_blob->cpu_data()));
        // if blob has not prv descriptor then set it to avoid conversions on next iterations
            this->_blob->set_prv_data_descriptor(this->get_shared_ptr(), true);     //Change from false to true, suggested by Czaja, Jacek
    } else {
        shared_ptr<MKLDNNMemoryDescriptor> blob_prv_mkldnn_mem_descr = get_mkldnn_prv_descriptor(this->_blob);

        //if (*blob_prv_mkldnn_mem_descr->prv_memory_pd() !=  *this->prv_memory_pd() || blob_prv_mkldnn_mem_descr->get_scale() != this->get_scale()) {
        if (blob_prv_mkldnn_mem_descr->prv_memory_pd() !=  this->prv_memory_pd()) {
            // prv in blob and in this descrptor may have different layouts
            this->convert_from_extprv(blob_prv_mkldnn_mem_descr->aprimitive());
        } else {
            this->_blob->mutable_prv_data();
        }
    }
}

void MKLDNNMemoryDescriptor::sync_before_write(bool inplace)
{
    // TODO: need to optimize code
    if(!inplace) {
        this->_blob->set_prv_data_descriptor(this->get_shared_ptr(), this->conversion_needed() ? false : true);
    }
    //Fix me: this->conversion_needed() == false means diff/data is in the CPU, no need to set the prv_diff/data_descriptor
    /*
    if ((!inplace) && (this->conversion_needed())) {
        if (is_diff) {
            this->_blob->set_prv_diff_descriptor(this->get_shared_ptr(), false);
        } else {
            this->_blob->set_prv_data_descriptor(this->get_shared_ptr(), false);
        }
    }
    */
}

shared_ptr<primitive> MKLDNNMemoryDescriptor::create_input(Blob * blob, bool set_prv_ptr)
{
    shared_ptr<mkldnn::primitive> pres;
    if (this->conversion_needed()) {
        pres = this->get_blob_prv_primitive(blob, set_prv_ptr, false);
    } else {
        pres.reset(new memory(*this->usr_memory_pd(), const_cast<real_t*> (blob->cpu_data())));
    }
    return pres;
}

shared_ptr<memory> MKLDNNMemoryDescriptor::create_output_memory(Blob * blob, bool inplace)
{
    shared_ptr<memory> omem;
    if (this->conversion_needed()) {
        shared_ptr<PrvMemDescr> blob_prv_mem_descriptor = blob->get_prv_data_descriptor();

        if(blob_prv_mem_descriptor != NULL) {
            shared_ptr<MKLDNNMemoryDescriptor > current_descr = get_mkldnn_prv_descriptor(blob);

            omem = current_descr->get_prv_memory();
            this->set_prv_memory(omem);
        } else {
            omem = this->get_prv_memory();
        }
    } else {
        omem.reset(new memory(*this->usr_memory_pd(), blob->mutable_cpu_data()));
    }
    return omem;
}

void MKLDNNMemoryDescriptorBase::convert_from_other(shared_ptr<PrvMemDescr> other)
{
    NOT_IMPLEMENTED;
}

bool MKLDNNMemoryDescriptorBase::layout_compare(shared_ptr<PrvMemDescr> other)
{
    CHECK_EQ(other->get_descr_type(),
              PrvMemDescr::PRV_DESCR_MKLDNN);

    shared_ptr<MKLDNNMemoryDescriptorBase > other_descr =
        std::static_pointer_cast<MKLDNNMemoryDescriptorBase >(other);

    return (*other_descr->prv_memory_pd() == *this->prv_memory_pd());
}

void MKLDNNMemoryDescriptorBase::check_usr_with_prv_descriptors()
{
    CHECK(_usr_memory_pd);
    CHECK(_prv_memory_pd);
    int32_t ndims = _usr_memory_pd->desc().data.ndims;
    CHECK_EQ(ndims, _prv_memory_pd->desc().data.ndims)
            << "MKLDNNMemoryDescriptorBase: Usr and Prv memory must have same dimensions number";
    for (int32_t dim = 0; dim < ndims; ++dim) {
        CHECK_EQ(_usr_memory_pd->desc().data.dims[dim]
                , _prv_memory_pd->desc().data.dims[dim])
                << "MKLDNNMemoryDescriptorBase: Usr and Prv memory must have same dimensions";
    }
}

shared_ptr<primitive> MKLDNNMemoryDescriptor::create_input(bool set_prv_ptr)
{
    // TODO: need to iptimize code
    return create_input(this->_blob, set_prv_ptr);
}


shared_ptr<memory> MKLDNNMemoryDescriptor::create_output_memory(bool inplace)
{
    // TODO: need to optimize code
    shared_ptr<memory> omem = create_output_memory(this->_blob);
    if(!inplace) {
        this->_blob->set_prv_data_descriptor(this->get_shared_ptr(), this->conversion_needed() ? false : true);
    }
    /*
    //Fix me: this->conversion_needed() == false means diff/data is in the CPU, no need to set the prv_diff/data_descriptor
    if ((!inplace) && (this->conversion_needed())) {
        if (is_diff) {
            this->_blob->set_prv_diff_descriptor(this->get_shared_ptr(), false);
        } else {
            this->_blob->set_prv_data_descriptor(this->get_shared_ptr(), false);
        }
    }
    */
    return omem;
}

real_t* MKLDNNMemoryDescriptor::get_memory_ptr(long offset) {
    if (this->conversion_needed()) {
      // TODO: support DFP16 offset
      if (this->prv_ptr() != NULL) return (real_t*)this->prv_ptr() + offset;
      // when _internal_ptr is null, having same private layout as _blob
      else return  (real_t*)this->_blob->prv_data() + offset;
    } else {
      return const_cast<real_t*>(this->_blob->cpu_data() + offset);
    }
}

shared_ptr<memory::desc> MKLDNNMemoryDescriptor::get_memory_desc() {
    shared_ptr<memory::desc> desc;
    if (this->conversion_needed()) {
        desc.reset(new memory::desc(this->prv_memory_pd()->desc()));
    } else {
        desc.reset(new memory::desc(this->usr_memory_pd()->desc()));
    }
    return desc;
}

size_t MKLDNNMemoryDescriptor::get_memory_count() {
  if (this->conversion_needed()) {
    return this->prv_count();
  } else {
    return this->_blob->count();
  }
}


void MKLDNNMemoryDescriptorBase::create_reorder_descriptors(std::vector<float> scale, int mask, std::vector<float> scale_ext, bool is_sum, bool is_wino)
{
    CHECK(_usr_memory_pd);
    CHECK(_prv_memory_pd);

    primitive_attr attri;
    int count = scale.size();
    if ( *_usr_memory_pd != *_prv_memory_pd) {
        std::vector<float> scales_u2p(count);
        #pragma omp parallel for if (count > 1)
        for(int i=0; i < count; i++){
            scales_u2p[i] = scale[i];
        }
        attri.set_output_scales(mask, scales_u2p);
        attri.set_int_output_round_mode(round_nearest);
        _reorder_usr2prv_pd = shared_ptr<reorder::primitive_desc>(
                new reorder::primitive_desc(*_usr_memory_pd, *_prv_memory_pd, attri));

        std::vector<float> scales_p2u(count);
        #pragma omp parallel for if (count > 1)
        for(int i=0; i < count; i++){
            scales_p2u[i] = (1. / scale[i]);
        }
        attri.set_output_scales(mask, scales_p2u);
        attri.set_int_output_round_mode(round_nearest);
        if(!is_wino)
            _reorder_prv2usr_pd = shared_ptr<reorder::primitive_desc>(
                    new reorder::primitive_desc(*_prv_memory_pd, *_usr_memory_pd, attri));
    }
    if ( _extprv_memory_pd && (*_prv_memory_pd != *_extprv_memory_pd || scale != scale_ext)) {
        if(is_sum == true && scale == scale_ext && _extprv_memory_pd->desc().data.data_type == memory::data_type::s8 && _prv_memory_pd->desc().data.data_type == memory::data_type::u8){
            _reorder_extprv2prv_pd = NULL;
        }else{
            std::vector<float> scales_e2p(count);
            float shift_scale;
            #pragma omp parallel for if (count > 1)
            for(int i=0; i < count; i++){
                shift_scale = scale[i] / scale_ext[i]; //fp32->int8 blob_prv_mkldnn_mem_descr->get_scale() will always be 0 ?
                scales_e2p[i] = shift_scale;
            }
            attri.set_output_scales(mask, scales_e2p);
            attri.set_int_output_round_mode(round_nearest);
            _reorder_extprv2prv_pd = shared_ptr<reorder::primitive_desc>(new reorder::primitive_desc(*_extprv_memory_pd, *_prv_memory_pd, attri));

        }
    }
}
}  // namespace caffe
#endif  // #ifdef MKLDNN_SUPPORTED