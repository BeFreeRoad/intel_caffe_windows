#ifdef USE_MKLDNN
#include <algorithm>
#include <vector>
#include "mkldnn_layers.hpp"
#include "engine_parser.hpp"
#include "mkldnn_memory.hpp"

namespace caffe {

void MKLDNNReLULayer::LayerSetUp(const vector<Blob*>& bottom
                                        ,const vector<Blob*>& top)
{
    VLOG(1) << "MKLDNNReLULayer::LayerSetUp: " << this->layer_param_.name();
    std::cout << "MKLDNNReLULayer::LayerSetUp: " << this->layer_param_.name();

    NeuronLayer::LayerSetUp(bottom, top);
}

void MKLDNNReLULayer::Reshape(const vector<Blob*>& bottom
                                    ,const vector<Blob*>& top)
{
    VLOG(1) << "MKLDNNReLULayer::Reshape: " << this->layer_param_.name();

    NeuronLayer::Reshape(bottom, top);

    this->reshape = (this->width_ == bottom[0]->width() &&
                     this->height_ == bottom[0]->height() &&
                     this->channels_ == bottom[0]->channels() &&
                     this->num_ == bottom[0]->num()) ? false : true;
    this->width_ = bottom[0]->width();
    this->height_ = bottom[0]->height();
    this->num_ = bottom[0]->num();
    this->channels_ = bottom[0]->channels();

}

void MKLDNNReLULayer::InitReLUFwd(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
    auto propagation = prop_kind::forward_scoring;
    int32_t n  = this->num_;
    int32_t iw = this->width_;
    int32_t ih = this->height_;
    int32_t ic = this->channels_;

    real_t negative_slope = this->layer_param_.relu_param().negative_slope();
    bool bottom_data_is_prv = (const_cast<real_t*>(bottom[0]->prv_data()) != NULL);
    bool inplace = (bottom[0] == top[0]);

    engine cpu_engine = CpuEngine::Instance().get_engine();
    memory::data_type mpcsn = memory::data_type::f32;
    // ---- Initialize memory descriptors -------------
    shared_ptr<memory::desc> bottom_data_md;
    shared_ptr<memory::primitive_desc> usr_data_mpd(NULL), prv_data_mpd(NULL), top_data_mpd(NULL);
    memory::data_type src_dt = memory::data_type::f32;
    memory::data_type top_dt = memory::data_type::f32;
    memory::format src_mfmt = memory::format::nchw;
    //bottom_data_is_prv = false;
    std::vector<float> scale;
    if (bottom_data_is_prv) {
        shared_ptr<MKLDNNMemoryDescriptor > mem_descr
            = get_mkldnn_prv_descriptor(bottom[0]);
        bottom_data_md.reset(new memory::desc(mem_descr->prv_memory_pd()->desc()));
        usr_data_mpd = mem_descr->usr_memory_pd();
        prv_data_mpd = mem_descr->prv_memory_pd();
        scale.push_back(mem_descr->get_scale(0));
        src_dt = static_cast<memory::data_type>(mem_descr->prv_memory_pd()->desc().data.data_type);
        src_mfmt = static_cast<memory::format>(mem_descr->prv_memory_pd()->desc().data.format);
    } else {
        bottom_data_md.reset(new memory::desc({{n, ic, ih, iw}}, mpcsn, memory::format::nchw));
        usr_data_mpd.reset(new memory::primitive_desc(*bottom_data_md, cpu_engine));
        prv_data_mpd.reset(new memory::primitive_desc(*bottom_data_md, cpu_engine));
        scale.push_back(1.);
    }
    top_dt = src_dt;
    top_data_mpd.reset(new memory::primitive_desc({{n,ic,ih,iw}, top_dt, src_mfmt}, cpu_engine));

    // ---- Initialize relu primitive descriptor -------------
    //relu_forward::desc reluFwd_desc(propagation, *bottom_data_md, negative_slope);
    // MKLDNN is deprecating standalone relu primitive in MKL-DNN.
    // Now MKLDNN has eltwise primitive with eltwise_relu algorithm inside.
    eltwise_forward::desc eltwise_reluFwd_desc(propagation, eltwise_relu, *bottom_data_md, negative_slope);

    // ---- Determining engine to use -----------------------
    std::string subengines = "MKLDNN:CPU";
    EngineParser ep(subengines);
    unsigned subEngineIndex = 0;
    reluFwd_pd = NULL;
    for(; subEngineIndex < ep.getNumberOfSubEngines(); subEngineIndex++) {
      try {
        reluFwd_pd.reset(new relu_forward::primitive_desc(eltwise_reluFwd_desc,
                ep.getMKLDNNSubEngine(subEngineIndex)));
      }
      catch(...) {
        continue;
      }
      break;
    }
    CHECK(reluFwd_pd);

    // ---  init primitive and prv_memory descriptors ----------------------
    fwd_bottom_data.reset(new MKLDNNData(usr_data_mpd, prv_data_mpd, bottom[0], this, scale));
    fwd_bottom_data->name = "fwd_bottom_data   @ " + this->layer_param_.name();
    fwd_bottom_data_primitive = fwd_bottom_data->create_input(false);

    fwd_top_data.reset(new MKLDNNData(usr_data_mpd, top_data_mpd, top[0], this, scale));
    fwd_top_data->name = "fwd_top_data   @ " + this->layer_param_.name();

    fwd_top_data_memory = fwd_top_data->create_output_memory(inplace);

    reluFwd.reset(new relu_forward(*reluFwd_pd, *fwd_bottom_data_primitive, *fwd_top_data_memory));
    //fwd_bottom_data->set_mkldnn_primitive(reluFwd);     //Wrong passed primitive! (TODO: Checking!)
    MKLDNNPrimitive fwd_bottom_data_primitive_transfer(fwd_bottom_data_primitive);
    fwd_bottom_data->set_mkldnn_primitive(fwd_bottom_data_primitive_transfer);

    //fwd_top_data->set_mkldnn_primitive(reluFwd);        //Wrong passed primitive! (TODO: Checking!)
    MKLDNNPrimitive fwd_top_data_memory_transfer(fwd_top_data_memory);
    fwd_top_data->set_mkldnn_primitive(fwd_top_data_memory_transfer);
}


void MKLDNNReLULayer::Forward_cpu(const vector<Blob*>& bottom
                                        ,const vector<Blob*>& top)
{
//    VLOG(1) << "MKLDNNReLULayer::Forward_cpu: " << this->layer_param_.name();
    std::cout << "MKLDNNReLULayer::Forward_cpu: " << this->layer_param_.name()<<std::endl;
    bool inplace = (bottom[0] == top[0]);
    if( reluFwd_pd == NULL || this->reshape)
        InitReLUFwd(bottom, top);

    // making reorders if needed.
    fwd_bottom_data->sync_before_read();
    // update top that head at prv
    fwd_top_data->sync_before_write(inplace);

    reluFwd.submit();
}
}  // namespace caffe
#endif  // #ifdef MKLDNN_SUPPORTED
