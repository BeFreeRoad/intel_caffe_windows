#ifdef USE_MKLDNN
#include <algorithm>
#include <cstdlib>
#include <vector>
#include "mkldnn_layers.hpp"
#include "engine_parser.hpp"

namespace caffe {
MKLDNNInnerProductLayer::MKLDNNInnerProductLayer(
            const LayerParameter& param) :
            MKLDNNLayer(param),
            InnerProductLayer(param),
            fwd_bottom_data(NULL),
            fwd_top_data(NULL),
            fwd_weights_data(NULL),
            fwd_bias_data(NULL),
            ipFwd_pd(NULL),
            fwd_top_data_memory(NULL),
            fwd_bottom_data_primitive(NULL),
            fwd_weights_data_primitive(NULL),
            fwd_bias_data_primitive(NULL),
            w_(0),
            h_(0)
{
  this->M_ = 0;
  this->K_ = 0;
}

MKLDNNInnerProductLayer::~MKLDNNInnerProductLayer()
{
}

void MKLDNNInnerProductLayer::LayerSetUp(const vector<Blob*>& bottom
                                            , const vector<Blob*>& top)
{
    VLOG(1) << "MKLDNNInnerProductLayer::LayerSetUp: " << this->layer_param_.name();
    InnerProductLayer::LayerSetUp(bottom, top);
}

void MKLDNNInnerProductLayer::Reshape(const vector<Blob*>& bottom
                                            , const vector<Blob*>& top)
{
    VLOG(1) << "MKLDNNInnerProductLayer::Reshape: " << this->layer_param_.name();
    const int axis = bottom[0]->CanonicalAxisIndex(
        this->layer_param_.inner_product_param().axis());
    if (this->M_ != bottom[0]->count(0, axis) ||
        this->K_ != bottom[0]->count(axis) ||
        this->w_ != bottom[0]->width() ||
        this->h_ != bottom[0]->height()) {
      this->reshape = true;
    } else {
      this->reshape = false;
    }

    InnerProductLayer::Reshape(bottom, top);

    this->w_ = bottom[0]->width();
    this->h_ = bottom[0]->height();
}

void MKLDNNInnerProductLayer::InitInnerProductFwd(const vector<Blob*>& bottom
                                                    , const vector<Blob*>& top)
{
    auto propagation = prop_kind::forward_scoring;

    int32_t n  = this->M_;
    int32_t w = this->w_;
    int32_t h = this->h_;
    int32_t oc = this->N_;
    int32_t ic = this->K_/h_/w_;
    bool has_spatial = (bottom[0]->shape().size() != 2);

    // Initialize memory descriptors (fromat = any) to create inner_product descriptor
    memory::data_type mpcsn = memory::data_type::f32;
    memory::format mfmt = memory::format::any;

    memory::dims bottom_tz = (has_spatial) ? memory::dims{n, ic, h, w} : memory::dims{n, ic};
    memory::dims top_tz = {n, oc};
    memory::dims weights_tz = (has_spatial) ? memory::dims {oc, ic, h, w} : memory::dims{oc, ic};
    memory::dims bias_tz = {oc};

#ifdef DEBUG
    if (has_spatial)
    {
        LOG(INFO) << "Dimension of bottom for MKLDNN: " << n << " " << ic << " " << h << " " << w;
        LOG(INFO) << "Dimension of weights for MKLDNN: " << oc << " " << ic << " " << h << " " << w;
    }
    else
    {
        LOG(INFO) << "Dimension of bottom for MKLDNN: " << n << " " << ic;
        LOG(INFO) << "Dimension of weights for MKLDNN: " << oc << " " << ic;
    }
#endif

    memory::desc init_bottom_md({bottom_tz}, mpcsn, mfmt);
    memory::desc init_top_md({top_tz}, mpcsn, mfmt);
    memory::desc init_weights_md({weights_tz}, mpcsn, mfmt);
    memory::desc init_bias_md({bias_tz}, mpcsn, mfmt);

    // Initialize inner_product primitive descriptor
    shared_ptr<inner_product_forward::desc> ipFwd_desc;
 
    if (this->bias_term_) {
        ipFwd_desc.reset(new inner_product_forward::desc(propagation, init_bottom_md, init_weights_md
                                                ,init_bias_md, init_top_md));
     } else {
        ipFwd_desc.reset(new inner_product_forward::desc(propagation, init_bottom_md, init_weights_md
                                                , init_top_md));
    }

    // ---- Determining engine to use -----------------------
//    std::string subengines = this->layer_param_.engine();
//    if (subengines.find("MKLDNN") == std::string::npos || subengines == "MKLDNN")
    std::string subengines = "MKLDNN:CPU";
    EngineParser ep(subengines);
    unsigned subEngineIndex = 0;
    ipFwd_pd = NULL;
    for(; subEngineIndex < ep.getNumberOfSubEngines(); subEngineIndex++) {
      try {
        ipFwd_pd.reset(new inner_product_forward::primitive_desc(*ipFwd_desc,
                ep.getMKLDNNSubEngine(subEngineIndex)));
      }
      catch(...) {
        continue;
      }
      break;
    }

    CHECK(ipFwd_pd);

    // Create priv memory primitive descriptors stored as class members
    typedef typename memory::primitive_desc MemPD; // short name for memory::primitive_desc

    shared_ptr<MemPD> prv_fwd_bottom_data_memory_pd(new MemPD(ipFwd_pd->src_primitive_desc()));
    shared_ptr<MemPD> prv_fwd_top_data_memory_pd(new MemPD(ipFwd_pd->dst_primitive_desc()));
    shared_ptr<MemPD> prv_fwd_weights_data_memory_pd(new MemPD(ipFwd_pd->weights_primitive_desc()));
 
    // Create usr memory primitive descriptors stored as class members
    engine cpu_engine = CpuEngine::Instance().get_engine();
    memory::format input_mfmt = has_spatial ? memory::format::nchw : memory::format::nc;
    shared_ptr<MemPD> usr_bottom_data_memory_pd(new MemPD({{bottom_tz}, mpcsn, input_mfmt}, cpu_engine));
    shared_ptr<MemPD> usr_bias_data_memory_pd(new MemPD({{bias_tz}, mpcsn, memory::format::x}, cpu_engine));
    shared_ptr<MemPD> usr_top_data_memory_pd(new MemPD({{top_tz}, mpcsn, memory::format::nc}, cpu_engine));
    memory::format weights_mfmt = has_spatial ? memory::format::oihw : memory::format::oi;
    shared_ptr<MemPD> usr_weights_data_memory_pd(new MemPD({{weights_tz}, mpcsn, weights_mfmt}, cpu_engine));
#ifdef DEBUG
    LOG(INFO) << "Memory format of usr_bottom_data_memory_pd: " << input_mfmt;
    LOG(INFO) << "Memory format of usr_weights_data_memory_pd: " << weights_mfmt;
#endif

    // ---  init primitive and prv_memory descriptors ----------------------
    fwd_bottom_data.reset(new MKLDNNData(usr_bottom_data_memory_pd, prv_fwd_bottom_data_memory_pd, bottom[0], this));
    fwd_bottom_data ->name = "fwd_bottom_data   @ " + this->layer_param_.name();
    fwd_bottom_data_primitive = fwd_bottom_data->create_input(false);

    fwd_top_data.reset(new MKLDNNData(usr_top_data_memory_pd, prv_fwd_top_data_memory_pd, top[0], this));
    fwd_top_data    ->name = "fwd_top_data      @ " + this->layer_param_.name();
    fwd_top_data_memory = fwd_top_data->create_output_memory();

    if (fwd_weights_data == NULL) {
      fwd_weights_data.reset(new MKLDNNData(usr_weights_data_memory_pd, prv_fwd_weights_data_memory_pd, this->blobs_[0].get(), this));
      fwd_weights_data->name = "fwd_weights_data  @ " + this->layer_param_.name();
      fwd_weights_data_primitive = fwd_weights_data->create_input(true);
    }

    if (this->bias_term_) {
        if (fwd_bias_data == NULL) {
          shared_ptr<MemPD> prv_fwd_bias_data_memory_pd(new MemPD(ipFwd_pd->bias_primitive_desc()));
          fwd_bias_data.reset(new MKLDNNData(usr_bias_data_memory_pd, prv_fwd_bias_data_memory_pd, this->blobs_[1].get(), this));
          fwd_bias_data   ->name = "fwd_bias_data     @ " + this->layer_param_.name();
          fwd_bias_data_primitive = fwd_bias_data->create_input(true);
        }
        ipFwd.reset(new inner_product_forward(*ipFwd_pd
                            , *fwd_bottom_data_primitive, *fwd_weights_data_primitive
                            , *fwd_bias_data_primitive, *fwd_top_data_memory));
    } else {
        ipFwd.reset(new inner_product_forward(*ipFwd_pd
                            , *fwd_bottom_data_primitive, *fwd_weights_data_primitive
                            , *fwd_top_data_memory));
    }
    
    //Because the inputs of inner product layer always come from user memory, so will not trigger the wrong reorder from extprv to prv
    //fwd_bottom_data->set_mkldnn_primitive(ipFwd);     //Wrong passed primitive! (TODO: Checking!)
    MKLDNNPrimitive fwd_bottom_data_primitive_transfer(fwd_bottom_data_primitive);
    fwd_bottom_data->set_mkldnn_primitive(fwd_bottom_data_primitive_transfer);

    //fwd_top_data->set_mkldnn_primitive(ipFwd);        //Wrong passed primitive! (TODO: Checking!)
    MKLDNNPrimitive fwd_top_data_memory_transfer(fwd_top_data_memory);
    fwd_top_data->set_mkldnn_primitive(fwd_top_data_memory_transfer);

    //fwd_weights_data->set_mkldnn_primitive(ipFwd);    //Wrong passed primitive! (TODO: Checking!)
    MKLDNNPrimitive fwd_weights_data_primitive_transfer(fwd_weights_data_primitive);
    fwd_weights_data->set_mkldnn_primitive(fwd_weights_data_primitive_transfer);

    if (this->bias_term_)
    {
      //fwd_bias_data->set_mkldnn_primitive(ipFwd);       //Wrong passed primitive! (TODO: Checking!)
      MKLDNNPrimitive fwd_bias_data_primitive_transfer(fwd_bias_data_primitive);
      fwd_bias_data->set_mkldnn_primitive(fwd_bias_data_primitive_transfer);
    }
}

void MKLDNNInnerProductLayer::Forward_cpu(const vector<Blob*>& bottom
                                                , const vector<Blob*>& top)
{
//    VLOG(1) << "MKLDNNInnerProductLayer::Forward_cpu: " << this->layer_param_.name();
#ifdef DEBUG
    LOG(INFO) << "MKLDNNInnerProductLayer::Forward_cpu: " << this->layer_param_.name();
#endif

    if( ipFwd_pd == NULL || this->reshape)
        InitInnerProductFwd(bottom, top);
    // making reorders if needed.
    fwd_bottom_data->sync_before_read();
    fwd_weights_data->sync_before_read();
    if (this->bias_term_)
      fwd_bias_data->sync_before_read();
    // update top that head at prv
    fwd_top_data->sync_before_write();

    ipFwd.submit();
}
}  // namespace caffe
#endif  // #ifdef MKLDNN_SUPPORTED
