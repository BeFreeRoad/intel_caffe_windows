#ifdef USE_MKLDNN
#include <algorithm>
#include <vector>

#include "mkldnn_layers.hpp"

namespace caffe {

void MKLDNNEltwiseLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
    Layer::LayerSetUp(bottom, top);

    CHECK(this->layer_param().eltwise_param().coeff_size() == 0
        || this->layer_param().eltwise_param().coeff_size() == bottom.size()) <<
        "Eltwise Layer takes one coefficient per bottom blob.";
    CHECK(!(this->layer_param().eltwise_param().operation() == EltwiseParameter_EltwiseOp_PROD
        && this->layer_param().eltwise_param().coeff_size())) <<
        "Eltwise layer only takes coefficients for summation.";
    op_ = this->layer_param_.eltwise_param().operation();
    // Blob-wise coefficients for the elementwise operation.
    coeffs_ = vector<real_t >(bottom.size(), 1);
    if (this->layer_param().eltwise_param().coeff_size())
    {
        for (int i = 0; i < bottom.size(); ++i) 
        {
            coeffs_[i] = this->layer_param().eltwise_param().coeff(i);
        }
    }
    num_bottoms_ = bottom.size();
    stable_prod_grad_ = this->layer_param_.eltwise_param().stable_prod_grad();
}

void MKLDNNEltwiseLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
    VLOG(1) << "MKLDNNEltwiseLayer::Reshape: " << this->layer_param_.name();
    this->reshape = (this->width_ == bottom[0]->width() &&
                     this->height_ == bottom[0]->height() &&
                     this->channels_ == bottom[0]->channels() &&
                     this->num_ == bottom[0]->num()) ? false : true;

    this->width_ = bottom[0]->width();
    this->height_ = bottom[0]->height();
    this->num_ = bottom[0]->num();
    this->channels_ = bottom[0]->channels();

    switch (op_)
    {
    case EltwiseParameter_EltwiseOp_PROD:
        NOT_IMPLEMENTED;
        break;
    case EltwiseParameter_EltwiseOp_SUM:
        {
            for (int i = 1; i < num_bottoms_; ++i)
            {
                CHECK(bottom[i]->shape() == bottom[0]->shape());
            }
            top[0]->ReshapeLike(*bottom[0]);
        }
        break;
    case EltwiseParameter_EltwiseOp_MAX:
        NOT_IMPLEMENTED;
        /*
        {
            for (int i = 1; i < num_bottoms_; ++i)
            {
                CHECK(bottom[i]->shape() == bottom[0]->shape());
            }
            top[0]->ReshapeLike(*bottom[0]);
            // If max operation, we will initialize the vector index part.
            if (this->layer_param_.eltwise_param().operation() == EltwiseParameter_EltwiseOp_MAX && top.size() == 1)
            {
                max_idx_.Reshape(bottom[0]->shape());
            }
        }
        */
        break;
    default:
        LOG(FATAL) << "Unknown elementwise operation.";
    }
}

void MKLDNNEltwiseLayer::InitEltwiseFwd(const vector<Blob*>& bottom, const vector<Blob*>& top)
{

    int32_t n  = this->num_;
    int32_t iw = this->width_;
    int32_t ih = this->height_;
    int32_t ic = this->channels_;

    // If we just do simple adding, scale is 1.0 for all inputs we have
    std::vector<float> scale(num_bottoms_, 1.0);
    //Eltwise layer is supporting multiplication coefficient and this scale value can be used for that.
    for (int i = 0; i < num_bottoms_; ++i) 
    {
        scale[i] = coeffs_[i];
    }

    engine cpu_engine = CpuEngine::Instance().get_engine();
    memory::data_type mpcsn = memory::data_type::f32;
    memory::format mfmt_nchw = memory::format::nchw;

    // ---- Initialize memory descriptors -------------
    std::vector<memory::data_type> prv_dt(num_bottoms_, memory::data_type::f32);
    for (auto i = 0; i < num_bottoms_; i++)
    {
        bool bottom_data_is_prv = (const_cast<real_t*>(bottom[i]->prv_data()) != NULL);
        if (bottom_data_is_prv)
        {
            shared_ptr<MKLDNNMemoryDescriptor > mem_descr
                = get_mkldnn_prv_descriptor(bottom[i]);
            prv_dt[i] = static_cast<memory::data_type>(mem_descr->prv_memory_pd()->desc().data.data_type);
        }
    }

    memory::data_type first_prv_dt = prv_dt[0];
    bool different_input_dt = false;
    for (auto i = 0; i < prv_dt.size(); i++)
    {
        if (prv_dt[i] != first_prv_dt)
        {
            different_input_dt = true;
        }
    }
    
    std::vector<memory::primitive_desc> bottom_data_mpd;
    fwd_bottom_data.clear();
    fwd_bottom_data_primitives_.clear();
    fwd_bottom_data_primitives_at_.clear();
    memory::data_type bottom_data_dt = memory::data_type::f32;
    for (auto i = 0; i < num_bottoms_; i++) 
    {
        fwd_bottom_data.push_back(shared_ptr<MKLDNNData >());
        memory::format bottom_data_mfmt = mfmt_nchw;
        shared_ptr<memory::primitive_desc> prv_bottom_data_mpd;
        shared_ptr<memory::primitive_desc> usr_bottom_data_mpd(
            new memory::primitive_desc({{n, ic, ih, iw}, mpcsn, mfmt_nchw}, cpu_engine));

        bool bottom_data_is_prv = (const_cast<real_t*>(bottom[i]->prv_data()) != NULL);
        if (bottom_data_is_prv)
        {
            shared_ptr<MKLDNNMemoryDescriptor > mem_descr
                = get_mkldnn_prv_descriptor(bottom[i]);
            if(!different_input_dt){
                bottom_data_mfmt = static_cast<memory::format>(
                    mem_descr->prv_memory_pd()->desc().data.format);
                bottom_data_dt = static_cast<memory::data_type>(
                    mem_descr->prv_memory_pd()->desc().data.data_type);
            }
            prv_bottom_data_mpd.reset(new memory::primitive_desc(
              {{n, ic, ih, iw}, bottom_data_dt, bottom_data_mfmt}, cpu_engine));
        }

        bottom_data_mpd.push_back(memory::primitive_desc(
            {{n, ic, ih, iw}, bottom_data_dt, bottom_data_mfmt}, cpu_engine));

        fwd_bottom_data[i].reset(new MKLDNNData(
            usr_bottom_data_mpd, prv_bottom_data_mpd, bottom[i], this));        
        fwd_bottom_data[i]->name = "fwd_bottom_data[i]   @ " + this->layer_param_.name();
        fwd_bottom_data_primitives_.push_back(fwd_bottom_data[i]->create_input(false));
        fwd_bottom_data_primitives_at_.push_back(*fwd_bottom_data_primitives_[i]);
    }

    shared_ptr<memory::primitive_desc> usr_top_data_mpd(new memory::primitive_desc(
        {{n, ic, ih, iw}, mpcsn, mfmt_nchw}, cpu_engine));
    
    // ---- Determining engine to use -----------------------
    std::string subengines = "MKLDNN:CPU";
    eltwiseFwd_pd.reset(new sum::primitive_desc({{n, ic, ih, iw}, bottom_data_dt, memory::format::any}, scale, bottom_data_mpd));
    CHECK(eltwiseFwd_pd);

    shared_ptr<memory::primitive_desc> prv_top_data_mpd(new memory::primitive_desc(eltwiseFwd_pd->dst_primitive_desc()));

    fwd_top_data.reset(new MKLDNNData(usr_top_data_mpd, prv_top_data_mpd, top[0], this));
    fwd_top_data->name = "fwd_top_data   @ " + this->layer_param_.name();
    fwd_top_data_memory = fwd_top_data->create_output_memory();

    eltwiseFwd.reset(new sum(*eltwiseFwd_pd, fwd_bottom_data_primitives_at_, *fwd_top_data_memory));
    
    for (auto i = 0; i < num_bottoms_; i++)
    {
        //fwd_bottom_data[i]->set_mkldnn_primitive(eltwiseFwd);   //Wrong passed primitive! (TODO: Checking!)
        MKLDNNPrimitive fwd_bottom_data_primitive_transfer(fwd_bottom_data_primitives_[i]);
        fwd_bottom_data[i]->set_mkldnn_primitive(fwd_bottom_data_primitive_transfer);
    }
    //fwd_top_data->set_mkldnn_primitive(eltwiseFwd);             //Wrong passed primitive! (TODO: Checking!)
    MKLDNNPrimitive fwd_top_data_memory_transfer(fwd_top_data_memory);
    fwd_top_data->set_mkldnn_primitive(fwd_top_data_memory_transfer);
}


void MKLDNNEltwiseLayer::Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
//    VLOG(1) << "MKLDNNEltwiseLayer::Forward_cpu: " << this->layer_param_.name();

    if(eltwiseFwd_pd == NULL || this->reshape)
        InitEltwiseFwd(bottom, top);
    for (auto i = 0; i < num_bottoms_; i++)
    {
        // making reorders if needed.
        fwd_bottom_data[i]->sync_before_read();
    }
    // update top that head at prv
    fwd_top_data->sync_before_write();

    eltwiseFwd.submit();
}

}  // namespace caffe
#endif  // #ifdef MKLDNN_SUPPORTED
