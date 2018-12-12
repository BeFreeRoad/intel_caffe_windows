#ifdef USE_MKLDNN
#include <algorithm>
#include <cfloat>
#include <vector>

#include "mkldnn_layers.hpp"
#include "../../util/math_functions.hpp"

namespace caffe {

void MKLDNNConcatLayer::LayerSetUp(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  VLOG(1) << "MKLDNNConcatLayer::LayerSetUp: " << this->layer_param_.name();

  const ConcatParameter& concat_param = this->layer_param_.concat_param();
  CHECK(!(concat_param.has_axis() && concat_param.has_concat_dim()))
      << "Either axis or concat_dim should be specified; not both.";

  int dim_src = bottom[0]->shape().size();
  //  int dim_dst = dim_src;

  num_concats_ = bottom.size();

  const int num_axes = bottom[0]->num_axes();
  if (concat_param.has_concat_dim()) {
    concat_dimension = static_cast<int>(concat_param.concat_dim());
    // Don't allow negative indexing for concat_dim, a uint32 -- almost certainly unintended.
    CHECK_GE(concat_dimension, 0) << "casting concat_dim from uint32 to int32 "
        << "produced negative result; concat_dim must satisfy "
        << "0 <= concat_dimension < " << kMaxBlobAxes;
    CHECK_LT(concat_dimension, num_axes) << "concat_dimension out of range.";
  } else {
    concat_dimension = bottom[0]->CanonicalAxisIndex(concat_param.axis());
  }

  for (auto i = 1; i < num_concats_; ++i) {
    if (concat_dimension == 0)
    {
      CHECK_EQ(bottom[0]->channels(), bottom[i]->channels());
      CHECK_EQ(bottom[0]->height(), bottom[i]->height());
      CHECK_EQ(bottom[0]->width(), bottom[i]->width());
      break;
    }
    else if (concat_dimension == 1)
    {
      CHECK_EQ(bottom[0]->num(), bottom[i]->num());
      if (!concat_param.per_fla_fuse()){
        CHECK_EQ(bottom[0]->height(), bottom[i]->height());
        CHECK_EQ(bottom[0]->width(), bottom[i]->width());
      }
      break;
    }
    else if (concat_dimension == 2)
    {
      CHECK_EQ(bottom[0]->num(), bottom[i]->num());
      CHECK_EQ(bottom[0]->channels(), bottom[i]->channels());
      CHECK_EQ(bottom[0]->width(), bottom[i]->width());
      break;
    }
    else if (concat_dimension == 3)
    {
      CHECK_EQ(bottom[0]->num(), bottom[i]->num());
      CHECK_EQ(bottom[0]->channels(), bottom[i]->channels());
      CHECK_EQ(bottom[0]->height(), bottom[i]->height());
      break;
    }
  }

  split_dims.reserve(num_concats_);
  if (concat_dimension == 0)
  {
    num_ = 0;
    channels_ = bottom[0]->channels();
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    for (auto i = 0; i < num_concats_; ++i) {
      CHECK_EQ(dim_src, bottom[i]->shape().size());
      split_dims[i] = bottom[i]->num();
      num_ += split_dims[i];
    }
  }
  else if (concat_dimension == 1)
  {
    num_ = bottom[0]->num();
    channels_ = 0;
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    if (concat_param.per_fla_fuse()){
      height_ = 1;
      width_ = 1;
      for (auto i = 0; i < num_concats_; ++i) {
        CHECK_EQ(dim_src, bottom[i]->shape().size());
        split_dims[i] = bottom[i]->channels()*bottom[i]->height()*bottom[i]->width();
        channels_ += split_dims[i];
      }
    } else{
      for (auto i = 0; i < num_concats_; ++i) {
        CHECK_EQ(dim_src, bottom[i]->shape().size());
        split_dims[i] = bottom[i]->channels();
        channels_ += split_dims[i];
      }
    }
  }
  else if (concat_dimension == 2)
  {
    num_ = bottom[0]->num();
    channels_ = bottom[0]->channels();
    height_ = 0;
    width_ = bottom[0]->width();
    for (auto i = 0; i < num_concats_; ++i) {
      CHECK_EQ(dim_src, bottom[i]->shape().size());
      split_dims[i] = bottom[i]->height();
      height_ += split_dims[i];
    }
  }
  else if (concat_dimension == 3)
  {
    num_ = bottom[0]->num();
    channels_ = bottom[0]->channels();
    height_ = bottom[0]->height();
    width_ = 0;
    for (auto i = 0; i < num_concats_; ++i) {
      CHECK_EQ(dim_src, bottom[i]->shape().size());
      split_dims[i] = bottom[i]->width();
      width_ += split_dims[i];
    }
  }
}

void MKLDNNConcatLayer::Reshape(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  VLOG(1) << "MKLDNNConcatLayer::Reshape: "  << this->layer_param_.name();
  const ConcatParameter& concat_param = this->layer_param_.concat_param();
  if (concat_dimension == 0)
  {
    //Need to re-calculate the shape duo to the change of batch size
    num_ = 0;
    channels_ = bottom[0]->channels();
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    //Also need to reshape the concat dim, in case the concat dim is just be reshaped by batch size
    for (auto i = 0; i < num_concats_; ++i) {
        split_dims[i] = bottom[i]->num();
        num_ += split_dims[i];
    }

    if (this->channels_ == bottom[0]->channels() &&
        this->height_ == bottom[0]->height() &&
        this->width_ == bottom[0]->width()) {
      this->reshape = false;
    } else {
      this->reshape = true;
    }
  }
  else if (concat_dimension == 1)
  {
    num_ = bottom[0]->num();
    channels_ = 0;
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    if (concat_param.per_fla_fuse()){
      height_ = 1;
      width_ = 1;
      for (auto i = 0; i < num_concats_; ++i) {
          split_dims[i] = bottom[i]->channels()*bottom[i]->height()*bottom[i]->width();
          channels_ += split_dims[i];
      }
      if (this->num_ == bottom[0]->num()) { 
        this->reshape = false;
      } else {
        this->reshape = true;
      }

    } else{
      for (auto i = 0; i < num_concats_; ++i) {
        split_dims[i] = bottom[i]->channels();
        channels_ += split_dims[i];
      }
      if (this->num_ == bottom[0]->num() &&
          this->height_ == bottom[0]->height() &&
          this->width_ == bottom[0]->width()) {
        this->reshape = false;
      } else {
        this->reshape = true;
      }
    }
  }
  else if (concat_dimension == 2)
  {
    num_ = bottom[0]->num();
    channels_ = bottom[0]->channels();
    height_ = 0;
    width_ = bottom[0]->width();
    for (auto i = 0; i < num_concats_; ++i) {
        split_dims[i] = bottom[i]->height();
        height_ += split_dims[i];
    }

    if (this->num_ == bottom[0]->num() &&
        this->channels_ == bottom[0]->channels() &&
        this->width_ == bottom[0]->width()) {
      this->reshape = false;
    } else {
      this->reshape = true;
    }
  }
  else if (concat_dimension == 3)
  {
    num_ = bottom[0]->num();
    channels_ = bottom[0]->channels();
    height_ = bottom[0]->height();
    width_ = 0;
    for (auto i = 0; i < num_concats_; ++i) {
        split_dims[i] = bottom[i]->width();
        width_ += split_dims[i];
    }

    if (this->num_ == bottom[0]->num() &&
        this->channels_ == bottom[0]->channels() &&
        this->height_ == bottom[0]->height()) {
      this->reshape = false;
    } else {
      this->reshape = true;
    }
  }

  top[0]->Reshape(num_, channels_, height_, width_);
}

void MKLDNNConcatLayer::InitConcatFwd(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {

  //Fix: MKLDNN concat layer should use 4D blob as input! Reshape the 2D input blob into 4D for calculation!
  bool has_spatial = (bottom[0]->shape().size() != 2);
#ifdef DEBUG
  LOG(INFO) << "has_spatial flag value: " << has_spatial;
#endif
  if (has_spatial == false)
  {
#ifdef DEBUG
      LOG(INFO) << "size of bottom blob: " << bottom[0]->shape().size();
      LOG(INFO) << "size of top blob: " << top[0]->shape().size();
      LOG(INFO) << "MKLDNN concat layer only support 4D blob as input! Reshape the 2D input blob into 4D for calculation!";
#endif
      for (auto i = 0; i < num_concats_; i++)
      {
          vector<int> bottom_4D_shape;
          int bottom_4D_height = 1;
          int bottom_4D_width = 1;
          bottom_4D_shape.push_back(bottom[i]->num());
          bottom_4D_shape.push_back(bottom[i]->channels());
          bottom_4D_shape.push_back(bottom_4D_height);
          bottom_4D_shape.push_back(bottom_4D_width);
          bottom[i]->Reshape(bottom_4D_shape, false);
      }      
  }
  engine cpu_engine = CpuEngine::Instance().get_engine();
  memory::data_type usr_dt = memory::data_type::f32;
  memory::data_type prv_dt = usr_dt;
  // memory::format mfmt_any = memory::format::any;
  memory::format mfmt_nchw = memory::format::nchw;

  memory::dims output_tz = {num_, channels_, height_, width_};
  std::vector<memory::primitive_desc> srcs_mpd;
  std::vector<primitive::at> srcs;
  fwd_bottom_data.clear();
  fwd_input_primitives_.clear();
  fwd_input_primitives_at_.clear();

  float scale = 1.;
  float scale_min = 1.;
  for (auto i = 0; i < num_concats_; i++) {
      if (const_cast<real_t *>(bottom[i]->prv_data()) != NULL) {
          shared_ptr<MKLDNNMemoryDescriptor > mem_descr
            = get_mkldnn_prv_descriptor(bottom[i]);
          scale = mem_descr->get_scale(0);
          if (scale_min == 1.) scale_min = scale;
          if(scale != 1. && scale < scale_min) scale_min = scale;
      }
  }
  std::vector<memory::format> src_mfmts;
  std::vector<float> bottom_scales(num_concats_, 1.);
  std::vector<shared_ptr<MKLDNNMemoryDescriptor >> mem_descr;

  std::vector<memory::data_type> prv_dt_tmp(num_concats_, memory::data_type::f32);
  bool different_input_dt = false;
  for(auto i = 0; i < num_concats_; i++) {
    if (const_cast<real_t *>(bottom[i]->prv_data()) != NULL) {
      shared_ptr<MKLDNNMemoryDescriptor > mem_descr_tmp = get_mkldnn_prv_descriptor(bottom[i]);
      prv_dt_tmp[i] = static_cast<memory::data_type>(mem_descr_tmp->prv_memory_pd()->desc().data.data_type);
    }
  }

  memory::data_type first_prv_dt = prv_dt_tmp[0];
  for (auto i = 0; i < prv_dt_tmp.size(); i++){
    if (prv_dt_tmp[i] != first_prv_dt){
      different_input_dt = true;
    }
  }

  for (auto i = 0; i < num_concats_; i++) {
    fwd_bottom_data.push_back(shared_ptr<MKLDNNData >());
    mem_descr.push_back(shared_ptr<MKLDNNMemoryDescriptor>());

    memory::dims input_tz = {0, 0, 0, 0};
    if (concat_dimension == 0)
    {
      input_tz = {split_dims[i], channels_, height_, width_};
    }
    else if (concat_dimension == 1)
    {
      input_tz = {num_, split_dims[i], height_, width_};
    }
    else if (concat_dimension == 2)
    {
      input_tz = {num_, channels_, split_dims[i], width_};
    }
    else if (concat_dimension == 3)
    {
      input_tz = {num_, channels_, height_, split_dims[i]};
    }

    memory::format src_mfmt = mfmt_nchw;
    shared_ptr<memory::primitive_desc> prv_src_mpd;
    shared_ptr<memory::primitive_desc> usr_src_mpd(
        new memory::primitive_desc({input_tz, usr_dt, mfmt_nchw}, cpu_engine));
 
    if (const_cast<real_t *>(bottom[i]->prv_data()) != NULL) {
      scale = 1.;
      mem_descr[i]  = get_mkldnn_prv_descriptor(bottom[i]);
      if(!different_input_dt){
        src_mfmt = static_cast<memory::format>(
            mem_descr[i]->prv_memory_pd()->desc().data.format);
        prv_dt = static_cast<memory::data_type>(mem_descr[i]->prv_memory_pd()->desc().data.data_type);
        scale = mem_descr[i]->get_scale(0);
        bottom_scales[i] = scale;
        if(scale != 1.) scale = scale_min;
      } 
      prv_src_mpd.reset(new memory::primitive_desc(
            {input_tz, prv_dt, src_mfmt}, cpu_engine));
    }
    std::vector<float> scale_bottom;
    scale_bottom.push_back(scale);

    src_mfmts.push_back(src_mfmt);
    srcs_mpd.push_back(memory::primitive_desc(
          {input_tz, prv_dt, src_mfmt}, cpu_engine));

    fwd_bottom_data[i].reset(new MKLDNNData(
          usr_src_mpd, prv_src_mpd, bottom[i], this, scale_bottom));
    fwd_input_primitives_.push_back(fwd_bottom_data[i]->create_input(false));
    fwd_input_primitives_at_.push_back(*fwd_input_primitives_[i]);
  }

  shared_ptr<memory::primitive_desc> usr_dst_mpd(new memory::primitive_desc(
        {output_tz, usr_dt, mfmt_nchw}, cpu_engine));

  concatFwd_pd.reset(new concat::primitive_desc(concat_dimension, srcs_mpd));

  shared_ptr<memory::primitive_desc> prv_dst_mpd(new memory::primitive_desc(
        concatFwd_pd->dst_primitive_desc()));

  std::vector<float> scale_top;
  if(!different_input_dt){
    scale_top.push_back(scale_min);
  } else{
    scale_top.push_back(1.);
  }
  fwd_top_data.reset(new MKLDNNData(usr_dst_mpd, prv_dst_mpd, top[0],
        this, scale_top));
 
  fwd_output_memory = fwd_top_data->create_output_memory();

  memory::format base_mfmt = mfmt_nchw;
  float base_scale = 1.;
  this->in_place_ = true;

  for(auto i = 0; i < num_concats_; i++){
    if(i == 0) {
      base_mfmt = src_mfmts[i];
      base_scale = bottom_scales[i];
    }
    else if((concat_dimension != 0 && bottom[i]->shape()[concat_dimension - 1] != 1) || base_mfmt != src_mfmts[i] || fabs(base_scale-bottom_scales[i]) > FLT_MIN || different_input_dt) {
      this->in_place_ = false;
      break;
    }
  }

  if(this->in_place_) {
    size_t offset = 0;     
    for(auto i = 0; i < num_concats_; i++){
      if(bottom[i]->prv_data()){
        if (scale_top[0] != 1.){
          memcpy(static_cast<char*>(fwd_output_memory->get_data_handle()) + offset, static_cast<char*>(mem_descr[i]->get_prv_memory()->get_data_handle()), sizeof(char) * bottom[i]->count());
	  mem_descr[i]->get_prv_memory()->set_data_handle(static_cast<char*>(fwd_output_memory->get_data_handle())+offset);
        }else{
          caffe_copy(bottom[i]->count(), static_cast<real_t*>(mem_descr[i]->get_prv_memory()->get_data_handle()), static_cast<real_t*>(fwd_output_memory->get_data_handle()) + offset);
          mem_descr[i]->get_prv_memory()->set_data_handle(static_cast<real_t*>(fwd_output_memory->get_data_handle()) + offset);
        }
      } else{
        caffe_copy(bottom[i]->count(), bottom[i]->cpu_data(), static_cast<real_t*>(fwd_output_memory->get_data_handle()) + offset);
        bottom[i]->set_cpu_data(static_cast<real_t*>(fwd_output_memory->get_data_handle()) + offset);
      }
      offset += bottom[i]->count();
    }
  }

  concatFwd.reset(new concat(*concatFwd_pd, fwd_input_primitives_at_, *fwd_output_memory));

  for (auto i = 0; i < num_concats_; i++) {
    //fwd_bottom_data[i]->set_mkldnn_primitive(concatFwd);  //Wrong passed primitive! (TODO: Checking!)
    MKLDNNPrimitive fwd_bottom_data_primitive_transfer(fwd_input_primitives_[i]);
    fwd_bottom_data[i]->set_mkldnn_primitive(fwd_bottom_data_primitive_transfer);
  }
  //fwd_top_data->set_mkldnn_primitive(concatFwd);          //Wrong passed primitive! (TODO: Checking!)
  MKLDNNPrimitive fwd_top_data_memory_transfer(fwd_output_memory);
  fwd_top_data->set_mkldnn_primitive(fwd_top_data_memory_transfer);
}

void MKLDNNConcatLayer::Forward_cpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  if ((NULL == concatFwd_pd) || (true == this->reshape))
    InitConcatFwd(bottom, top);

  for (auto i = 0; i < num_concats_; i++) {
    // making reorders if needed.
    fwd_bottom_data[i]->sync_before_read();
  }
    // update top that head at prv
    fwd_top_data->sync_before_write();

  if(!this->in_place_) {
    concatFwd.submit();
  }
}

} // namespace caffe

#endif
