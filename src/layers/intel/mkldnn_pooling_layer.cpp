#ifdef USE_MKLDNN
#include <algorithm>
#include <cfloat>
#include <vector>

#include "mkldnn_layers.hpp"
#include "engine_parser.hpp"

namespace caffe {

void MKLDNNPoolingLayer::LayerSetUp(const vector<Blob*>& bottom
                                            ,const vector<Blob*>& top)
{
    VLOG(1) << "MKLDNNPoolingLayer::LayerSetUp: " << this->layer_param_.name();

    Layer::LayerSetUp(bottom, top);
    PoolingParameter pool_param = this->layer_param_.pooling_param();

    if (pool_param.global_pooling()) {
        CHECK(!(pool_param.has_kernel_size() || pool_param.has_kernel_h() || pool_param.has_kernel_w()))
            << "With Global_pooling: true Filter size cannot specified";
    } else {
        CHECK(!pool_param.has_kernel_size() != !(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
            << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
        CHECK(pool_param.has_kernel_size() ||(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
            << "For non-square filters both kernel_h and kernel_w are required.";
    }
    CHECK((!pool_param.has_pad() && pool_param.has_pad_h() && pool_param.has_pad_w())
            || (!pool_param.has_pad_h() && !pool_param.has_pad_w()))
        << "pad is pad OR pad_h and pad_w are required.";
    CHECK((!pool_param.has_stride() && pool_param.has_stride_h() && pool_param.has_stride_w())
            || (!pool_param.has_stride_h() && !pool_param.has_stride_w()))
        << "Stride is stride OR stride_h and stride_w are required.";

    global_pooling_ = pool_param.global_pooling();
    if (global_pooling_) {
        kernel_h_ = bottom[0]->height();
        kernel_w_ = bottom[0]->width();
    } else {
        if (pool_param.has_kernel_size()) {
          kernel_h_ = kernel_w_ = pool_param.kernel_size();
        } else {
          kernel_h_ = pool_param.kernel_h();
          kernel_w_ = pool_param.kernel_w();
        }
    }

    CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
    CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";

    if (!pool_param.has_pad_h()) {
        pad_t_ = pad_b_ = pad_l_ = pad_r_ = pool_param.pad();
    } else {
        pad_t_ = pad_b_ = pool_param.pad_h();
        pad_l_ = pad_r_ = pool_param.pad_w();
    }

    if (!pool_param.has_stride_h()) {
        stride_h_ = stride_w_ = pool_param.stride();
    } else {
        stride_h_ = pool_param.stride_h();
        stride_w_ = pool_param.stride_w();
    }

    if (global_pooling_) {
        CHECK(pad_t_ == 0 && pad_l_ == 0 && stride_h_ == 1 && stride_w_ == 1)
            << "With Global_pooling: true; only pad = 0 and stride = 1";
    }
    if (pad_t_ != 0 || pad_l_ != 0) {
        CHECK(this->layer_param_.pooling_param().pool() == PoolingParameter_PoolMethod_AVE
        || this->layer_param_.pooling_param().pool() == PoolingParameter_PoolMethod_MAX)
        << "Padding implemented only for average and max pooling.";
        CHECK_LT(pad_t_, kernel_h_);
        CHECK_LT(pad_l_, kernel_w_);
    }
    compute_output_shape(bottom, top);
}

void MKLDNNPoolingLayer::compute_output_shape(const vector<Blob*>& bottom
                                        ,const vector<Blob*>& top)
{
    height_out_ = static_cast<int>(ceil(static_cast<float>(
        bottom[0]->height() + pad_t_ + pad_b_ - kernel_h_) / stride_h_)) + 1;
    width_out_ = static_cast<int>(ceil(static_cast<float>(
        bottom[0]->width() + pad_r_ + pad_l_ - kernel_w_) / stride_w_)) + 1;

    if (pad_t_ || pad_b_ || pad_r_ || pad_l_ || kernel_h_ == 1 || kernel_w_ == 1) {
        // If we have padding, ensure that the last pooling starts strictly
        // inside the image (instead of at the padding); otherwise clip the last.
        if ((height_out_ - 1) * stride_h_ >= bottom[0]->height() + pad_t_) {
          --height_out_;
        }
        if ((width_out_ - 1) * stride_w_ >= bottom[0]->width() + pad_l_) {
          --width_out_;
        }
        CHECK_LT((height_out_ - 1) * stride_h_, bottom[0]->height() + pad_t_);
        CHECK_LT((width_out_ - 1) * stride_w_, bottom[0]->width() + pad_l_);
    }
    else
    {
      // If user did not define padding, just use the exclude padding
      force_exclude_padding_flag_ = true;
    }

    //Add the pad to make sure h/w + kernel_h/w_ can be exact division by stride_h/w_
    auto h = bottom[0]->height() + pad_t_;
    while (h + pad_b_ < stride_h_ * (height_out_ - 1) + kernel_h_) pad_b_++;

    auto w = bottom[0]->width() + pad_l_;
    while (w + pad_r_ < stride_w_ * (width_out_ - 1) + kernel_w_) pad_r_++;
}

void MKLDNNPoolingLayer::Reshape(const vector<Blob*>& bottom
                                        ,const vector<Blob*>& top)
{
    VLOG(1) << "MKLDNNPoolingLayer::Reshape: "  << this->layer_param_.name();

    this->reshape = (this->width_ == bottom[0]->width() &&
                     this->height_ == bottom[0]->height() &&
                     this->channels_ == bottom[0]->channels() &&
                     this->num_ == bottom[0]->num()) ? false : true;
    this->num_ = bottom[0]->num();
    this->channels_ = bottom[0]->channels();
    this->height_ = bottom[0]->height();
    this->width_ = bottom[0]->width();

    compute_output_shape(bottom, top);

    CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
        << "corresponding to (num, channels, height, width)";

    top[0]->Reshape(bottom[0]->num(), channels_, height_out_, width_out_);

    if (top.size() > 1) {
        (reinterpret_cast<BlobInt* > (top[1]) )->Reshape(num_,
            channels_, height_out_, width_out_);
    }
    if (top.size() == 1) {
        max_idx_.Reshape(bottom[0]->num(), channels_, height_out_, width_out_);
    }
}

void MKLDNNPoolingLayer::InitPoolingFwd(const vector<Blob*>& bottom, const vector<Blob*>& top)
{

    auto propagation = prop_kind::forward_scoring;

    algorithm pooling_algorithm;
    switch (this->layer_param_.pooling_param().pool()) {
    case PoolingParameter_PoolMethod_MAX:
        pooling_algorithm = algorithm::pooling_max;
        break;
    case PoolingParameter_PoolMethod_AVE:
        if (this->layer_param_.pooling_param().avg_include_pad()) {
            pooling_algorithm = algorithm::pooling_avg_include_padding;
        }else {
            pooling_algorithm = algorithm::pooling_avg_exclude_padding;
        }
        // If user did not define padding
        // bottom[0]->height/width() + kernel_h/w_ cannot be exact division by stride_h/w_
        // use the exclude padding to align with the result of Caffe
        // for exact division situation, exclude padding and include padding will have the same results
        if (force_exclude_padding_flag_ == true)
        {
          pooling_algorithm = algorithm::pooling_avg_exclude_padding;
        }
        break;
    default:
        LOG(FATAL) << "Unknown pooling method.";
    }

    int32_t n = this->num_;
    int32_t c = this->channels_;
    int32_t ih = this->height_;
    int32_t iw = this->width_;
    int32_t oh = this->height_out_;
    int32_t ow = this->width_out_;

    int32_t kh = this->kernel_h_;
    int32_t kw = this->kernel_w_;

    int32_t sh = this->stride_h_;
    int32_t sw = this->stride_w_;

    int32_t pt = this->pad_t_;
    int32_t pb = this->pad_b_;
    int32_t pl = this->pad_l_;
    int32_t pr = this->pad_r_;

    bool bottom_data_is_prv = (const_cast<real_t *>(bottom[0]->prv_data()) != NULL);
    bool top_data_is_prv = (const_cast<real_t *>(top[0]->prv_data()) != NULL);

    engine cpu_engine = CpuEngine::Instance().get_engine();
    memory::data_type mpcsn = memory::data_type::f32;
    memory::dims bottom_tz = {n, c, ih, iw};
    memory::dims top_tz = {n, c, oh, ow};
    memory::format mfmt_nchw = memory::format::nchw;

    // ---- Initialize memory descriptors -------------
    typedef typename memory::primitive_desc MemPD; // short name for memory::primitive_desc
    memory::format cmfmt = mfmt_nchw;

    shared_ptr<MemPD> usr_bottom_data_mpd(new MemPD({{bottom_tz}, mpcsn, mfmt_nchw}, cpu_engine));
    shared_ptr<MemPD> usr_top_data_mpd(new MemPD({{top_tz}, mpcsn, mfmt_nchw}, cpu_engine));

    std::vector<float> scale;
    if (bottom_data_is_prv || top_data_is_prv) {
        shared_ptr<MKLDNNMemoryDescriptor > mem_descr
            = get_mkldnn_prv_descriptor(bottom[0]);
        cmfmt = static_cast<memory::format>(mem_descr->prv_memory_pd()->desc().data.format);
        mpcsn = static_cast<memory::data_type>(mem_descr->prv_memory_pd()->desc().data.data_type);
        scale.push_back(mem_descr->get_scale(0));
    } else{
        scale.push_back(1.);
    }

    shared_ptr<memory::desc> init_fwd_bottom_md(new memory::desc({bottom_tz}, mpcsn, cmfmt));
    shared_ptr<memory::desc> init_fwd_top_md(new memory::desc({top_tz}, mpcsn, cmfmt));

    // ---- Initialize pooling primitive descriptor -------------
    pooling_forward::desc poolingFwd_desc(propagation, pooling_algorithm, *init_fwd_bottom_md,*init_fwd_top_md
                                        , {sh, sw}, {kh, kw}, {pt, pl}, {pb, pr}, padding_kind::zero);
    // ---- Determining engine to use -----------------------
//    std::string subengines = this->layer_param_.engine();
//    if (subengines.find("MKLDNN") == std::string::npos || subengines == "MKLDNN")
    std::string subengines = "MKLDNN:CPU";
    EngineParser ep(subengines);
    unsigned subEngineIndex = 0;
    poolingFwd_pd = NULL;
    for(; subEngineIndex < ep.getNumberOfSubEngines(); subEngineIndex++) {
      try {
        poolingFwd_pd.reset(new pooling_forward::primitive_desc(poolingFwd_desc,
                ep.getMKLDNNSubEngine(subEngineIndex)));
      }
      catch(...) {
        continue;
      }
      break;
    }

    CHECK(poolingFwd_pd);
    engine engine = ep.getMKLDNNSubEngine(subEngineIndex);

    // ---- Initialize remaining memory descriptors -------------
    shared_ptr<MemPD> prv_fwd_bottom_data_mpd;
    shared_ptr<MemPD> prv_fwd_top_data_mpd;
    if (bottom_data_is_prv || top_data_is_prv) {
        prv_fwd_bottom_data_mpd.reset(new MemPD(*init_fwd_bottom_md, engine));
        prv_fwd_top_data_mpd.reset(new MemPD(*init_fwd_top_md, engine));
        // ---- Log prv memory primitive descriptors -------------
    }

    // ---- Create priv memory  ---------------------

    // We'll output the mask to top[1] if it's of size >1.
    int* mask = NULL;  // suppress warnings about uninitalized variables
    // We'll output the mask to top[1] if it's of size >1.
    const bool use_top_mask = top.size() > 1;
    mask = (use_top_mask) ?  reinterpret_cast<int*>(top[1]->mutable_cpu_data())
            : max_idx_.mutable_cpu_data();

    // ---  init primitive and prv_memory descriptors ----------------------
    fwd_bottom_data.reset(new MKLDNNData(usr_bottom_data_mpd, prv_fwd_bottom_data_mpd, bottom[0], this, scale));
    fwd_bottom_data_primitive = fwd_bottom_data->create_input(false);

    fwd_top_data.reset(new MKLDNNData(usr_top_data_mpd, prv_fwd_top_data_mpd, top[0], this, scale));
    fwd_top_data_memory = fwd_top_data->create_output_memory();

    if (propagation == prop_kind::forward_training &&
            pooling_algorithm != algorithm::pooling_avg_exclude_padding &&
            pooling_algorithm != algorithm::pooling_avg_include_padding) {
        indices_pd.reset(new MemPD(poolingFwd_pd->workspace_primitive_desc()));
        indices_memory.reset(new memory(*indices_pd, reinterpret_cast<void *>(mask)));
        poolingFwd.reset(new pooling_forward(*poolingFwd_pd, *fwd_bottom_data_primitive, *fwd_top_data_memory, *indices_memory));
    } else {
        poolingFwd.reset(new pooling_forward(*poolingFwd_pd, *fwd_bottom_data_primitive, *fwd_top_data_memory));
    }
    //fwd_bottom_data->set_mkldnn_primitive(poolingFwd);  //Wrong passed primitive! (TODO: Checking!)
    MKLDNNPrimitive fwd_bottom_data_primitive_transfer(fwd_bottom_data_primitive);
    fwd_bottom_data->set_mkldnn_primitive(fwd_bottom_data_primitive_transfer);

    //fwd_top_data->set_mkldnn_primitive(poolingFwd);     //Wrong passed primitive! (TODO: Checking!)
    MKLDNNPrimitive fwd_top_data_memory_transfer(fwd_top_data_memory);
    fwd_top_data->set_mkldnn_primitive(fwd_top_data_memory_transfer);
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
void MKLDNNPoolingLayer::Forward_cpu(const vector<Blob*>& bottom
                                            ,const vector<Blob*>& top)
{
    VLOG(1) << "MKLDNNPoolingLayer::Forward_cpu: " << this->layer_param_.name();
#ifdef DEBUG
    LOG(INFO) << "MKLDNNPoolingLayer::Forward_cpu: " << this->layer_param_.name();
#endif

    if (NULL == poolingFwd_pd || this->reshape)
        InitPoolingFwd(bottom, top);
    // making reorders if needed.
    fwd_bottom_data->sync_before_read();
    // update top that head at prv
    fwd_top_data->sync_before_write();

    poolingFwd.submit();
}
}  // namespace caffe
#endif  // #ifdef MKLDNN_SUPPORTED
