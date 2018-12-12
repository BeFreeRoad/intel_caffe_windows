#ifdef USE_MKLDNN

#include <algorithm>
#include <vector>
#include "engine_parser.hpp"
#include "mkldnn_layers.hpp"

namespace caffe {
MKLDNNConvolutionLayer::MKLDNNConvolutionLayer(const LayerParameter& param)
            : MKLDNNLayer(param), ConvolutionLayer(param)
            , fwd_bottom_data(NULL), fwd_top_data(NULL), fwd_weights_data(NULL), fwd_bias_data(NULL)
            , convFwd_pd(NULL)
            , fwd_top_data_memory(NULL)
            , fwd_bottom_data_primitive(NULL), fwd_weights_data_primitive(NULL), fwd_bias_data_primitive(NULL)
            , width_(0), height_(0), depth_(0), width_out_(0), height_out_(0), depth_out_(0), kernel_w_(0), kernel_h_(0), kernel_d_(0)
            , stride_w_(0), stride_h_(0), stride_d_(0), pad_w_(0), pad_h_(0), pad_d_(0)
{
}

void MKLDNNConvolutionLayer::compute_output_shape()
{
    ConvolutionLayer::compute_output_shape();
    CHECK_GT(this->output_shape_.size(), 1) << "MKLDNN Convolution layer expects at least 2D spatial output dimension!";
    CHECK_LT(this->output_shape_.size(), 4) << "MKLDNN Convolution layer expects at most 3D spatial output dimension!";
    if (this->output_shape_.size() == 2) {
      this->height_out_ = this->output_shape_[0];
      this->width_out_ = this->output_shape_[1];
    } else {
      this->depth_out_ = this->output_shape_[0];
      this->height_out_ = this->output_shape_[1];
      this->width_out_ = this->output_shape_[2];
    }
}

void MKLDNNConvolutionLayer::LayerSetUp(const vector<Blob*>& bottom
                                            , const vector<Blob*>& top)
{
    ConvolutionLayer::LayerSetUp(bottom, top);
    init_properties(bottom, top);
    this->bottom_shape_ = &bottom[0]->shape();

}

void MKLDNNConvolutionLayer::Reshape(const vector<Blob*>& bottom
                                            , const vector<Blob*>& top)
{
    if (this->num_spatial_axes_ == 2) {
      this->reshape = (this->width_ == bottom[0]->shape(3) &&
                       this->height_ == bottom[0]->shape(2) &&
                       this->channels_ == bottom[0]->shape(1) &&
                       this->num_ == bottom[0]->shape(0)) ? false : true;
    } else {
      this->reshape = (this->depth_ == bottom[0]->shape(2) &&
                       this->width_ == bottom[0]->shape(4) &&
                       this->height_ == bottom[0]->shape(3) &&
                       this->channels_ == bottom[0]->shape(1) &&
                       this->num_ == bottom[0]->shape(0)) ? false : true;
    }
    init_properties(bottom, top);
    BaseConvolutionLayer::ReshapeForMKL(bottom, top);
    // if (bottom.size() > 1) {
//    if (this->layer_param_.convolution_param().fusion_type() ==
//            ConvolutionParameter::SUM_FUSION &&
//        bottom.size() > 1) {
//      top[0]->ShareData(*bottom[1]);
//    }
}

void MKLDNNConvolutionLayer::init_properties(const vector<Blob*>& bottom
                                                , const vector<Blob*>& top)
{
    CHECK_GT(this->num_spatial_axes_, 1) << "MKLDNN Convolution layer expects at least 2D spatial input dimension!";
    CHECK_LT(this->num_spatial_axes_, 4) << "MKLDNN Convolution layer expects at most 3D spatial input dimension!";

    this->num_ = bottom[0]->shape(0);
    this->channels_ = bottom[0]->shape(1);

    if (this->num_spatial_axes_ == 2) {
      this->height_ = bottom[0]->shape(2);
      this->width_ = bottom[0]->shape(3);

      this->stride_h_ = this->stride_.cpu_data()[0];
      this->stride_w_ = this->stride_.cpu_data()[1];

      this->pad_h_ = this->pad_.cpu_data()[0];
      this->pad_w_ = this->pad_.cpu_data()[1];

      this->kernel_h_  = this->kernel_shape_.cpu_data()[0];
      this->kernel_w_ = this->kernel_shape_.cpu_data()[1];
    } else {
      this->depth_ = bottom[0]->shape(2);
      this->height_ = bottom[0]->shape(3);
      this->width_ = bottom[0]->shape(4);

      this->stride_d_ = this->stride_.cpu_data()[0];
      this->stride_h_ = this->stride_.cpu_data()[1];
      this->stride_w_ = this->stride_.cpu_data()[2];

      this->pad_d_ = this->pad_.cpu_data()[0];
      this->pad_h_ = this->pad_.cpu_data()[1];
      this->pad_w_ = this->pad_.cpu_data()[2];

      this->kernel_d_ = this->kernel_shape_.cpu_data()[0];
      this->kernel_h_  = this->kernel_shape_.cpu_data()[1];
      this->kernel_w_ = this->kernel_shape_.cpu_data()[2];
    }

    string _conv_algorithm = this->layer_param_.convolution_param().conv_algorithm();
    if(_conv_algorithm == "direct")
    {
        conv_algorithm = algorithm::convolution_direct;
    }
    else if(_conv_algorithm == "winograd")
    {
        conv_algorithm = algorithm::convolution_winograd;
    }
    else
    {
        LOG(ERROR) << "Unsupported convolution algorithm.";
        CHECK(false);
    }
}

void MKLDNNConvolutionLayer::InitConvolutionFwd(const vector<Blob*>& bottom
                                                , const vector<Blob*>& top)
{
    auto propagation = prop_kind::forward_scoring;
//    bool relu = this->layer_param_.convolution_param().relu();
//    real_t negative_slope = 0;
//    if(relu)
//    {
//        negative_slope = this->layer_param_.relu_param().negative_slope();
//    }
    int32_t g  = std::max(this->group_, 1);
    int32_t n  = this->num_;
    int32_t iw = this->width_;
    int32_t ih = this->height_;
    int32_t ic = this->channels_;
    int32_t id = this->depth_;

    int32_t ow = this->width_out_;
    int32_t oh = this->height_out_;
    int32_t oc = this->num_output_;
    int32_t od = this->depth_out_;

    int32_t kw = this->kernel_w_;
    int32_t kh = this->kernel_h_;
    int32_t kd = this->kernel_d_;

    int32_t sw = this->stride_w_;
    int32_t sh = this->stride_h_;
    int32_t sd = this->stride_d_;

    int32_t pw = this->pad_w_;
    int32_t ph = this->pad_h_;
    int32_t pd = this->pad_d_;

    memory::dims convolutionStrides;
    memory::dims padding;
    memory::dims padding_r;
    memory::dims dilation;
    bool dilated_conv = false;
    const int* dilation_data = this->dilation_.cpu_data();
    for (int i = 0; i < this->num_spatial_axes_; ++i) {
      dilation.push_back(dilation_data[i] - 1);
      if (dilation_data[i] != 1) dilated_conv = true;
    }

    if (this->num_spatial_axes_ == 2) {
      convolutionStrides = {sh, sw};
      padding = {ph, pw};
      padding_r.push_back((oh - 1) * sh - ih + ((kh - 1) * (dilation_data[0]) + 1) - ph);
      padding_r.push_back((ow - 1) * sw - iw + ((kw - 1) * (dilation_data[1]) + 1) - pw);
    } else {
      convolutionStrides = {sd, sh, sw};
      padding = {pd, ph, pw};
      padding_r.push_back((od - 1) * sd - id + ((kd - 1) * (dilation_data[0]) + 1) - pd);
      padding_r.push_back((oh - 1) * sh - ih + ((kh - 1) * (dilation_data[1]) + 1) - ph);
      padding_r.push_back((ow - 1) * sw - iw + ((kw - 1) * (dilation_data[2]) + 1) - pw);
    }

    // ---- Initialize memory descriptors (fromat = any) to create convolution descriptor -------------
    memory::data_type mpcsn = memory::data_type::f32;
    memory::data_type bottom_dt = memory::data_type::f32;
    memory::data_type top_dt = memory::data_type::f32;

    bool is_sum = false;
//    if (this->layer_param_.convolution_param().fusion_type() ==
//            ConvolutionParameter::SUM_FUSION &&
//        bottom.size() > 1) {
//      is_sum = true;
//
//      memory::data_type bottom_1_dt = memory::data_type::f32;
//      if (const_cast<real_t*>(bottom[1]->prv_data()) != NULL){
//
//        shared_ptr<MKLDNNMemoryDescriptor > bottom_1_desc =
//            get_mkldnn_prv_descriptor(bottom[1]);
//        bottom_1_dt = static_cast<memory::data_type>(bottom_1_desc->prv_memory_pd()->desc().data.data_type);
//      }
//
//      if (top_dt != bottom_1_dt) {
//        top_dt = bottom_1_dt;
//        // FIXME: to simplify the calibration tool to handle different data types of conv sum in residual block
//        if(top_dt ==  memory::data_type::f32){
////          this->need_quantize_ = false;
//          bottom_dt = memory::data_type::f32;
//        }
//      }
//    }


    memory::data_type weights_dt = memory::data_type::f32;
    memory::data_type bias_dt = memory::data_type::f32;
    memory::format mfmt_any = memory::format::any;

    memory::dims bottom_tz;
    memory::dims bias_tz;
    memory::dims top_tz;
    memory::dims weights_tz;
    if (this->num_spatial_axes_ == 2) {
      bottom_tz = {n, ic, ih, iw};
      bias_tz = {oc};
      top_tz = {n, oc, oh, ow};
      weights_tz = (g!= 1) ? memory::dims{g, oc/g, ic/g, kh, kw} : memory::dims{oc, ic, kh, kw};
    } else {
      bottom_tz = {n, ic, id, ih, iw};
      bias_tz = {oc};
      top_tz = {n, oc, od, oh, ow};
      weights_tz = (g!= 1) ? memory::dims{g, oc/g, ic/g, kd, kh, kw} : memory::dims{oc, ic, kd, kh, kw};
    }

    // ---- Memory descriptors for initializing of convolution primitive descriptor -------------
    memory::desc init_bottom_md({bottom_tz}, bottom_dt, mfmt_any);
    memory::desc init_bias_md({bias_tz}, bias_dt, mfmt_any);
    memory::desc init_top_md({top_tz}, top_dt, mfmt_any);
    memory::desc init_weights_md({weights_tz}, weights_dt, mfmt_any);

//    size_t coeff_size = this->layer_param_.convolution_param().coeff_size();
//    float coeff0 = 1;
//    float coeff1 = 1;
//    if (coeff_size == 2)
//    {
//      coeff0 = this->layer_param_.convolution_param().coeff(0);
//      coeff1 = this->layer_param_.convolution_param().coeff(1);
//    }

    primitive_attr attr;
    // ---- Determining engine to use -----------------------
    //std::string subengines = this->layer_param_.engine();
    //if (subengines.find("MKLDNN") == std::string::npos || subengines == "MKLDNN")
    std::string subengines = "MKLDNN:CPU";
    EngineParser ep(subengines);
    unsigned subEngineIndex = 0;
    mkldnn::algorithm eligibleAlgorithms[2] = {conv_algorithm, algorithm::convolution_direct};
    convFwd_pd = NULL;
    mkldnn::post_ops ops;
    float scale = 1.0f;
//    real_t alpha = negative_slope;  // negative slope for mkldnn_eltwise_relu.
    float beta = 1.0f;             // ignored for mkldnn_eltwise_relu.

//    if (this->layer_param_.convolution_param().fusion_type() ==
//            ConvolutionParameter::SUM_FUSION &&
//        bottom.size() > 1) {
//        ops.append_sum(1.0f);
//    }

//    if (relu) ops.append_eltwise(scale, eltwise_relu, alpha, beta);
    attr.set_post_ops(ops);

    for (auto& convAlgorithm : eligibleAlgorithms) {
      // ---- Initialize convolution primitive descriptor -------------
      shared_ptr<convolution_forward::desc> convFwd_desc;
      if (this->bias_term_) {
          if (dilated_conv)
              convFwd_desc.reset(new convolution_forward::desc(
                  propagation, convAlgorithm, init_bottom_md, init_weights_md,
                  init_bias_md, init_top_md, convolutionStrides, dilation, padding, padding_r,
                  padding_kind::zero));
          else
              convFwd_desc.reset(new convolution_forward::desc(
                  propagation, convAlgorithm, init_bottom_md, init_weights_md,
                  init_bias_md, init_top_md, convolutionStrides, padding, padding,
                  padding_kind::zero));
      } else {
          if (dilated_conv)
              convFwd_desc.reset(new convolution_forward::desc(
                  propagation, convAlgorithm, init_bottom_md, init_weights_md,
                  init_top_md, convolutionStrides, dilation, padding, padding_r,
                  padding_kind::zero));
          else
              convFwd_desc.reset(new convolution_forward::desc(
                  propagation, convAlgorithm, init_bottom_md, init_weights_md,
                  init_top_md, convolutionStrides, padding, padding,
                  padding_kind::zero));
      }

      for (subEngineIndex = 0; subEngineIndex < ep.getNumberOfSubEngines();
           subEngineIndex++) {
        try {
            convFwd_pd.reset(new convolution_forward::primitive_desc(
                *convFwd_desc, ep.getMKLDNNSubEngine(subEngineIndex)));

        } catch (...) {
            continue;
        }

        break;
      }
      if (convFwd_pd) break;
    }

    CHECK(convFwd_pd);
    engine cpu_engine = CpuEngine::Instance().get_engine();

    // ---- Create priv memory primitive descriptors stored as class members -------------
    typedef typename memory::primitive_desc MemPD; // short name for memory::primitive_desc

    shared_ptr<MemPD> prv_fwd_bottom_data_memory_pd(new MemPD(convFwd_pd->src_primitive_desc()));
    shared_ptr<MemPD> prv_fwd_top_data_memory_pd(new MemPD(convFwd_pd->dst_primitive_desc()));
    shared_ptr<MemPD> prv_fwd_weights_data_memory_pd(new MemPD(convFwd_pd->weights_primitive_desc()));

    // ---- Create usr memory primitive descriptors -------------
    memory::format data_mfmt;
    memory::format weights_mfmt;
    if (this->num_spatial_axes_ == 2) {
      data_mfmt = memory::format::nchw;
      weights_mfmt = (g!= 1) ? memory::format::goihw : memory::format::oihw;
    } else {
      data_mfmt = memory::format::ncdhw;
      weights_mfmt = (g!= 1) ? memory::format::goidhw : memory::format::oidhw;
    }

    shared_ptr<MemPD> usr_bottom_data_memory_pd(new MemPD({{bottom_tz}, mpcsn, data_mfmt}, cpu_engine));
    shared_ptr<MemPD> usr_bias_data_memory_pd(new MemPD({{bias_tz}, mpcsn, memory::format::x}, cpu_engine));
    shared_ptr<MemPD> usr_top_data_memory_pd(new MemPD({{top_tz}, mpcsn, data_mfmt}, cpu_engine));
    shared_ptr<MemPD> usr_weights_data_memory_pd(new MemPD({{weights_tz}, mpcsn, weights_mfmt}, cpu_engine));

    // ---  init primitive and prv_memory descriptors ----------------------
    fwd_bottom_data.reset(new MKLDNNData(usr_bottom_data_memory_pd, prv_fwd_bottom_data_memory_pd, bottom[0], this));
    fwd_bottom_data->name = "fwd_bottom_data   @ " + this->layer_param_.name();
    fwd_bottom_data_primitive = fwd_bottom_data->create_input(false);

    fwd_top_data.reset(new MKLDNNData(usr_top_data_memory_pd, prv_fwd_top_data_memory_pd, top[0], this));
    fwd_top_data->name = "fwd_top_data      @ " + this->layer_param_.name();
    fwd_top_data_memory = fwd_top_data->create_output_memory();

    bool is_wino = conv_algorithm == algorithm::convolution_winograd ? true : false;
    if (fwd_weights_data == NULL) {
      fwd_weights_data.reset(new MKLDNNData(usr_weights_data_memory_pd, prv_fwd_weights_data_memory_pd, this->blobs_[0].get(), this, {1.}, 0,  is_sum, is_wino));
      fwd_weights_data->name = "fwd_weights_data  @ " + this->layer_param_.name();
      fwd_weights_data_primitive = fwd_weights_data->create_input(true);
    }
    if (this->bias_term_) {
        if (fwd_bias_data == NULL) {
          shared_ptr<MemPD> prv_fwd_bias_data_memory_pd(new MemPD(convFwd_pd->bias_primitive_desc()));
          fwd_bias_data.reset(new MKLDNNData(usr_bias_data_memory_pd, prv_fwd_bias_data_memory_pd, this->blobs_[1].get(), this));
          fwd_bias_data->name = "fwd_bias_data     @ " + this->layer_param_.name();
          fwd_bias_data_primitive = fwd_bias_data->create_input(true);
        }
        convFwd.reset(new convolution_forward(*convFwd_pd
                        , *fwd_bottom_data_primitive, *fwd_weights_data_primitive
                        , *fwd_bias_data_primitive, *fwd_top_data_memory));
        //fwd_bias_data->set_mkldnn_primitive(convFwd);   //Wrong passed primitive! (For sure!)
        MKLDNNPrimitive fwd_bias_data_primitive_transfer(fwd_bias_data_primitive);
        fwd_bias_data->set_mkldnn_primitive(fwd_bias_data_primitive_transfer);
    } else {
        convFwd.reset(new convolution_forward(*convFwd_pd
                        , *fwd_bottom_data_primitive, *fwd_weights_data_primitive
                        , *fwd_top_data_memory));
    }
    MKLDNNPrimitive fwd_bottom_data_primitive_transfer(fwd_bottom_data_primitive);
    fwd_bottom_data->set_mkldnn_primitive(fwd_bottom_data_primitive_transfer);

    MKLDNNPrimitive fwd_top_data_memory_transfer(fwd_top_data_memory);
    fwd_top_data->set_mkldnn_primitive(fwd_top_data_memory_transfer);

    MKLDNNPrimitive fwd_weights_data_primitive_transfer(fwd_weights_data_primitive);
    fwd_weights_data->set_mkldnn_primitive(fwd_weights_data_primitive_transfer);
}

void MKLDNNConvolutionLayer::Forward_cpu(const std::vector<Blob*>& bottom
                                                , const std::vector<Blob*>& top)
{
    if( convFwd_pd == NULL || this->reshape)
        InitConvolutionFwd(bottom, top);
    fwd_bottom_data->sync_before_read();
    fwd_weights_data->sync_before_read();
    if (this->bias_term_)
        fwd_bias_data->sync_before_read();
    // update top that head at prv
    fwd_top_data->sync_before_write();

    convFwd.submit();
}

}   // namespace caffe

#endif  // USE_CUDNN
