#ifdef USE_MKLDNN
#include <algorithm>
#include <vector>

#include "mkldnn_layers.hpp"
#include "../../util/math_functions.hpp"
#include "engine_parser.hpp"

namespace caffe {

void MKLDNNBatchNormLayer::InitStatsBatchVars(int batch_size) {
    num_stats_batches_ = 1;
    stats_batch_size_ = batch_size;
    BatchNormParameter param = this->layer_param_.batch_norm_param();
    if (!use_global_stats_ && param.stats_batch_size() > 0) {
      CHECK_EQ(batch_size % param.stats_batch_size(), 0);
      num_stats_batches_ = batch_size / param.stats_batch_size();
      stats_batch_size_ = param.stats_batch_size();
    }
}

void MKLDNNBatchNormLayer::LayerSetUp(const vector<Blob*>& bottom
                                        ,const vector<Blob*>& top)
{
    VLOG(1) << "MKLDNNBatchNormLayer::LayerSetUp: " << this->layer_param_.name();

    Layer::LayerSetUp(bottom, top);

    channels_ = bottom[0]->channels();
    height_   = bottom[0]->height();
    width_    = bottom[0]->width();
    num_      = bottom[0]->num();

    eps_ = this->layer_param_.batch_norm_param().eps();
    use_weight_bias_ = this->layer_param_.batch_norm_param().use_weight_bias();
    bias_term_ = this->layer_param_.batch_norm_param().bias_term();
    moving_average_fraction_ = this->layer_param_.batch_norm_param().moving_average_fraction();
    use_global_stats_ = true;
//    if (this->layer_param_.batch_norm_param().has_use_global_stats())
//      use_global_stats_ = this->layer_param_.batch_norm_param().use_global_stats();

    InitStatsBatchVars(num_);

    this->blobs_.resize(3 + (use_weight_bias_ ? 1:0) + (use_weight_bias_ && bias_term_ ? 1:0));

    vector<int> sz;
    sz.push_back(channels_);
    this->blobs_[0].reset(new Blob(sz));
    this->blobs_[1].reset(new Blob(sz));
    sz[0]=1;
    this->blobs_[2].reset(new Blob(sz));
    for (int i = 0; i < 3; ++i) {
        caffe_set(this->blobs_[i]->count(), real_t (0),
            this->blobs_[i]->mutable_cpu_data());
    }

    //IntelCaffe treat scale and shift as different blobs, so current MKL-DNN integration has additional copies from Caffe to MKL-DNN buffer on fwd pass and from MKL-DNN to Caffe buffer on bwd pass.
    //Optimization: use the temp blob to combine the scale and shift together. Avoid the additional copies.
    // Initialize scale and shift combination blob
    vector<int> scaleshift_blob_shape(1);
    scaleshift_blob_shape[0] = 2*channels_;
    scaleshift_blob_.reset(new Blob(scaleshift_blob_shape));
    //Should initialize the scaleshift_blob_ buffer to 0, because when bias_term_ == false, need to pass zero bias to MKLDNN
    caffe_set(scaleshift_blob_shape[0], static_cast<real_t >(0),
              scaleshift_blob_->mutable_cpu_data());
    scaleshift_acc_ = scaleshift_blob_;
    if (num_stats_batches_ > 1) {
      this->scaleshift_acc_.reset(new Blob(scaleshift_blob_shape));
    }

    if (use_weight_bias_) {
        // Initialize scale and shift
        vector<int> scaleshift_shape(1);
        scaleshift_shape[0] = channels_;
        VLOG(1) << "MKLDNNBatchNormLayer::LayerSetUp: channels_  = " << channels_;

        this->blobs_[3].reset(new Blob(scaleshift_shape));
        this->blobs_[3]->set_cpu_data(scaleshift_blob_->mutable_cpu_data());

        if (bias_term_) {
            this->blobs_[4].reset(new Blob(scaleshift_shape));
            this->blobs_[4]->set_cpu_data(scaleshift_blob_->mutable_cpu_data() + scaleshift_blob_->offset(channels_));
        }
    }

    // Mask statistics from optimization by setting local learning rates
    // for mean, variance, and the bias correction to zero.
    for (int i = 0; i < 3; ++i) {
      if (this->layer_param_.param_size() == i) {
        ParamSpec* fixed_param_spec = this->layer_param_.add_param();
        fixed_param_spec->set_lr_mult(0.f);
      } else {
        CHECK_EQ(this->layer_param_.param(i).lr_mult(), 0.f)
            << "Cannot configure batch normalization statistics as layer "
            << "parameters.";
      }
    }
}

void MKLDNNBatchNormLayer::Reshape(const vector<Blob*>& bottom
                                    ,const vector<Blob*>& top)
{
    VLOG(1) << "MKLDNNBatchNormLayer::Reshape: " << this->layer_param_.name();

    this->reshape = (this->width_ == bottom[0]->width() &&
                     this->height_ == bottom[0]->height() &&
                     this->channels_ == bottom[0]->channels() &&
                     this->num_ == bottom[0]->num()) ? false : true;

    this->width_ = bottom[0]->width();
    this->height_ = bottom[0]->height();
    this->num_ = bottom[0]->num();
    this->channels_ = bottom[0]->channels();

    InitStatsBatchVars(this->num_);

    //Fix: should reshape the top blob with the real size of bottom blob
    //top[0]->Reshape(this->num_, this->channels_, this->height_, this->width_);
#ifdef DEBUG
    LOG(INFO) << "size of bottom blob: " << bottom[0]->shape().size();
#endif
    top[0]->ReshapeLike(*bottom[0]);
}

void MKLDNNBatchNormLayer::InitBatchNorm(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
    auto propagation = prop_kind::forward_scoring;

    unsigned flags = 0;
    if (use_weight_bias_) flags |= use_scale_shift;
    if (use_global_stats_) flags |= use_global_stats;

    int32_t n  = this->num_;
    int32_t iw = this->width_;
    int32_t ih = this->height_;
    int32_t ic = this->channels_;    

    bool bottom_data_is_prv = (const_cast<real_t *>(bottom[0]->prv_data()) != NULL);

    bool inplace = (bottom[0] == top[0]);
    engine cpu_engine = CpuEngine::Instance().get_engine();
    memory::data_type mpcsn = memory::data_type::f32;
    
    // ---- Initialize memory descriptors -------------
    shared_ptr<memory::desc> input_md, input_stats_md, output_md, scaleshift_md;
    shared_ptr<memory::primitive_desc> usr_mpd, prv_mpd;
    shared_ptr<memory::primitive_desc> scaleshift_mpd;
    if (bottom_data_is_prv) {
        shared_ptr<MKLDNNMemoryDescriptor > mem_descr
            = get_mkldnn_prv_descriptor(bottom[0]);
        input_md.reset(new memory::desc(mem_descr->prv_memory_pd()->desc()));
        usr_mpd = mem_descr->usr_memory_pd();
        prv_mpd = mem_descr->prv_memory_pd();
    } else {
        input_md.reset(new memory::desc({{n, ic, ih, iw}}, mpcsn, memory::format::nchw));   //MKLDNN batch norm only support 4D memory descriptor!
        usr_mpd.reset(new memory::primitive_desc(*input_md, cpu_engine));
    }
    output_md = input_md;
    input_stats_md.reset(new memory::desc(*input_md));
    CHECK(input_stats_md->data.ndims > 0 &&
          input_stats_md->data.dims[0] == this->num_);
    input_stats_md->data.dims[0] = stats_batch_size_;

    // ---- Initialize BatchNorm primitive descriptor -------------
    batch_normalization_forward::desc BatchNormFwd_desc(propagation, *input_stats_md, eps_, flags);
    // ---- Determining engine to use -----------------------
    std::string subengines = "MKLDNN:CPU";
    EngineParser ep(subengines);
    unsigned subEngineIndex = 0;
    BatchNormFwd_pd = NULL;
    bool relu = this->layer_param_.batch_norm_param().relu();
    mkldnn::primitive_attr attr;
    mkldnn::post_ops ops;
    if (relu) {
        ops.append_eltwise(1.f, eltwise_relu, 0.f, 0.f);
        attr.set_post_ops(ops);
    }
    for(; subEngineIndex < ep.getNumberOfSubEngines(); subEngineIndex++) {
      try {
        if (relu)
            BatchNormFwd_pd.reset(new batch_normalization_forward::primitive_desc(BatchNormFwd_desc, attr,
                ep.getMKLDNNSubEngine(subEngineIndex)));
        else
            BatchNormFwd_pd.reset(new batch_normalization_forward::primitive_desc(BatchNormFwd_desc,
                ep.getMKLDNNSubEngine(subEngineIndex)));
      }
      catch(...) {
        continue;
      }
      break;
    }

    CHECK(BatchNormFwd_pd);

    // ---- Create memory  ---------------------
    if (use_weight_bias_) {
        //For test in train, memory address of blobs_[3] and blobs_[4] will be changed when share data from train net. If the address
        // of blobs_[3] and blobs_[4] are continued, we will use them immediately, otherwise we will copy them to scaleshift_blob_ in Forward.
        if((this->blobs_[3]->mutable_cpu_data() + this->blobs_[3]->offset(channels_)) == this->blobs_[4]->mutable_cpu_data()){
            scaleshift_memory.reset(new memory(BatchNormFwd_pd->weights_primitive_desc(), this->blobs_[3]->mutable_cpu_data()));
        }else {
            scaleshift_memory.reset(new memory(BatchNormFwd_pd->weights_primitive_desc(), this->scaleshift_blob_->mutable_cpu_data()));
        }
    }

    // ---  init primitive and prv_memory descriptors ----------------------
    fwd_bottom_data.reset(new MKLDNNData(usr_mpd, prv_mpd, bottom[0], this));
    input_primitive = fwd_bottom_data->create_input(false);

    fwd_top_data.reset(new MKLDNNData(usr_mpd, prv_mpd, top[0], this));
    output_memory = fwd_top_data->create_output_memory();

    mean_memory.resize(num_stats_batches_);
    variance_memory.resize(num_stats_batches_);
    input_stats.resize(num_stats_batches_);
    output_stats.resize(num_stats_batches_);
    BatchNormFwd.resize(num_stats_batches_);
    for (int i = 0; i < num_stats_batches_; i++) {
      InitBatchNormFwdPrimitive(i);
    }

    //fwd_bottom_data->set_mkldnn_primitive(BatchNormFwd);  //Wrong passed primitive! (TODO: Checking!)
    MKLDNNPrimitive fwd_bottom_data_primitive_transfer(input_primitive);
    fwd_bottom_data->set_mkldnn_primitive(fwd_bottom_data_primitive_transfer);

    //fwd_top_data->set_mkldnn_primitive(BatchNormFwd);     //Wrong passed primitive! (TODO: Checking!)
    MKLDNNPrimitive fwd_top_data_memory_transfer(output_memory);
    fwd_top_data->set_mkldnn_primitive(fwd_top_data_memory_transfer);

    //Fix: MKLDNN batch norm only support 4D memory descriptor! Use 4D for calculation and reshape to 2D for output!
    bool has_spatial = (bottom[0]->shape().size() != 2);
#ifdef DEBUG
    LOG(INFO) << "has_spatial flag value: " << has_spatial;
#endif
    if (has_spatial == false)
    {
#ifdef DEBUG
        LOG(INFO) << "size of bottom blob: " << bottom[0]->shape().size();
        LOG(INFO) << "MKLDNN batch norm only support 4D memory descriptor! Use 4D for calculation and reshape to 2D for output!";
#endif
        vector<int> top_shape;
        top_shape.push_back(bottom[0]->num());
        top_shape.push_back(bottom[0]->channels());
        top[0]->Reshape(top_shape);
    }
}

shared_ptr<memory> MKLDNNBatchNormLayer::GetStatsBatchMemory(
  shared_ptr<MKLDNNMemoryDescriptor > mkldnn_mem, int idx) {
    long data_offset =
      idx * stats_batch_size_ * this->channels_ * this->width_ * this->height_;
    engine cpu_engine = CpuEngine::Instance().get_engine();
    shared_ptr<memory::desc> stats_md = mkldnn_mem->get_memory_desc();
    CHECK(stats_md->data.ndims > 0 &&
          stats_md->data.dims[0] == this->num_);
    stats_md->data.dims[0] = stats_batch_size_;
    shared_ptr<memory::primitive_desc> stats_mpd(
      new memory::primitive_desc(*stats_md, cpu_engine));
    shared_ptr<memory> stats(
      new memory(*stats_mpd, mkldnn_mem->get_memory_ptr(data_offset)));
    return stats;
}

void MKLDNNBatchNormLayer::InitBatchNormFwdPrimitive(int idx) {
    input_stats[idx] = GetStatsBatchMemory(fwd_bottom_data, idx);
    output_stats[idx] = GetStatsBatchMemory(fwd_top_data, idx);

    // ---- Create BatchNorm --------------------
//    if (this->phase_ == TEST && !use_global_stats_) {
    if (!use_global_stats_) {
        if (use_weight_bias_) {
            BatchNormFwd[idx].reset(new batch_normalization_forward(*BatchNormFwd_pd,
                    *input_stats[idx], *scaleshift_memory,
                    *output_stats[idx]));
        } else {
            BatchNormFwd[idx].reset(new batch_normalization_forward(*BatchNormFwd_pd,
                    *input_stats[idx], *output_stats[idx]));
        }
    } else {
        mean_memory[idx].reset(new memory(BatchNormFwd_pd->mean_primitive_desc()));
        variance_memory[idx].reset(new memory(BatchNormFwd_pd->variance_primitive_desc()));

        if (use_global_stats_) {
            caffe_copy(this->channels_, this->blobs_[0]->cpu_data(),
                static_cast<real_t *>(mean_memory[idx]->get_data_handle()));
            caffe_copy(this->channels_, this->blobs_[1]->cpu_data(),
               static_cast<real_t *>(variance_memory[idx]->get_data_handle()));
            if (use_weight_bias_) {
                BatchNormFwd[idx].reset(new batch_normalization_forward(*BatchNormFwd_pd,
                        *input_stats[idx], (const primitive::at)*mean_memory[idx],
                        (const primitive::at)*variance_memory[idx], *scaleshift_memory,
                        *output_stats[idx]));
            } else {
                BatchNormFwd[idx].reset(new batch_normalization_forward(*BatchNormFwd_pd,
                        *input_stats[idx], (const primitive::at)*mean_memory[idx],
                        (const primitive::at)*variance_memory[idx], *output_stats[idx]));
            }
        } else {
            if (use_weight_bias_) {
                BatchNormFwd[idx].reset(new batch_normalization_forward(*BatchNormFwd_pd,
                        *input_stats[idx], *scaleshift_memory, *output_stats[idx],
                        *mean_memory[idx], *variance_memory[idx]));
            } else {
                BatchNormFwd[idx].reset(new batch_normalization_forward(*BatchNormFwd_pd,
                        *input_stats[idx], *output_stats[idx], *mean_memory[idx], *variance_memory[idx]));
            }
        }
    }
}

void MKLDNNBatchNormLayer::Forward_cpu(const vector<Blob*>& bottom
                                        ,const vector<Blob*>& top)
{
    if(BatchNormFwd_pd == NULL || this->reshape)
        InitBatchNorm(bottom, top);
    bool inplace = (bottom[0] == top[0]);

    // making reorders if needed.
    fwd_bottom_data->sync_before_read();
    // update top that head at prv
    fwd_top_data->sync_before_write();

    for (int stats_batch_idx = 0; stats_batch_idx < num_stats_batches_; stats_batch_idx++) {
      if (use_global_stats_) {
        // use the stored mean/variance estimates.
        const real_t scale_factor = this->blobs_[2]->cpu_data()[0] == 0 ?
            0 : 1 / this->blobs_[2]->cpu_data()[0];
        real_t *mean_buffer_ = (real_t *)(mean_memory[stats_batch_idx]->get_data_handle());
        real_t *variance_buffer_ = (real_t *)(variance_memory[stats_batch_idx]->get_data_handle());

        //TODO: optimize, do this operation in the InitBatchNorm, so no need to calculate each time
        caffe_cpu_scale(this->blobs_[0]->count(), scale_factor,
                    this->blobs_[0]->cpu_data(), mean_buffer_);
        caffe_cpu_scale(this->blobs_[1]->count(), scale_factor,
                    this->blobs_[1]->cpu_data(), variance_buffer_);
      }
      
      BatchNormFwd[stats_batch_idx].submit();
    }
}
}  // namespace caffe
#endif  // #ifdef MKLDNN_SUPPORTED
