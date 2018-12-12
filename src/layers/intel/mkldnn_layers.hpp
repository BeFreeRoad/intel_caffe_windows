#ifndef CAFFE_MKLDNN_LAYERS_HPP_
#define CAFFE_MKLDNN_LAYERS_HPP_

#ifdef USE_MKLDNN
#include "../conv_layer.hpp"
#include "mkldnn_memory.hpp"
#include "mkldnn.hpp"
#include "mkldnn_base.hpp"
#include "../neuron_layer.hpp"
#include "../inner_product_layer.hpp"

namespace caffe {
//class MKLDNNLayer : public BaseQuantLayer {
class MKLDNNLayer {
public:
    explicit MKLDNNLayer(const LayerParameter &param){};
    virtual ~MKLDNNLayer() {}
protected:
    bool reshape;
};

class MKLDNNConvolutionLayer : public MKLDNNLayer , public ConvolutionLayer {
public:
    explicit MKLDNNConvolutionLayer(const LayerParameter& param);
    virtual ~MKLDNNConvolutionLayer() {}

    //For test the parameters of kernel/stride/pad
    int GetKernelWidth()  { return kernel_w_; }
    int GetKernelHeight() { return kernel_h_; }
    int GetKernelDepth()  { return kernel_d_; }

    int GetStrideWidth()  { return stride_w_; }
    int GetStrideHeight() { return stride_h_; }
    int GetStrideDepth()  { return stride_d_; }

    int GetPadWidth()     { return pad_w_; }
    int GetPadHeight()    { return pad_h_; }
    int GetPadDepth()     { return pad_d_; }
protected:
    virtual void Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
    // Customized methods
    virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
    void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);
private:
    virtual void compute_output_shape();
    virtual void init_properties(const vector<Blob*>& bottom, const vector<Blob*>& top);
    void InitConvolutionFwd(const vector<Blob*>& bottom, const vector<Blob*>& top);

    shared_ptr<MKLDNNData > fwd_bottom_data, fwd_top_data, fwd_weights_data, fwd_bias_data;
    shared_ptr<convolution_forward::primitive_desc> convFwd_pd;
    MKLDNNPrimitive convFwd;
    shared_ptr<memory> fwd_top_data_memory;
    shared_ptr<primitive> fwd_bottom_data_primitive, fwd_weights_data_primitive, fwd_bias_data_primitive;
    int32_t width_, height_, depth_, width_out_, height_out_, depth_out_, kernel_w_, kernel_h_, kernel_d_, stride_w_, stride_h_, stride_d_;
    int  pad_w_, pad_h_, pad_d_;
    mkldnn::algorithm  conv_algorithm;

};

class MKLDNNReLULayer : public MKLDNNLayer , public NeuronLayer  {
public:
    /**
    * @param param provides ReLUParameter relu_param,
    *     with ReLULayer options:
    *   - negative_slope (\b optional, default 0).
    *     the value @f$ \nu @f$ by which negative values are multiplied.
    */
  explicit MKLDNNReLULayer(const LayerParameter& param)
    : MKLDNNLayer(param), NeuronLayer(param)
    , fwd_top_data(), fwd_bottom_data()
    , reluFwd_pd()
    , fwd_top_data_memory()
    , fwd_bottom_data_primitive()
    , num_(0), width_(0), height_(0), channels_(0) {}
  ~MKLDNNReLULayer() {}

protected:
    virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
    virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);
    virtual inline const char* type() const { return "ReLU"; }
    virtual void Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
private:
    void InitReLUFwd(const vector<Blob*>& bottom, const vector<Blob*>& top);

    shared_ptr<MKLDNNData > fwd_top_data, fwd_bottom_data;
    shared_ptr<relu_forward::primitive_desc> reluFwd_pd;
    MKLDNNPrimitive reluFwd;
    shared_ptr<memory> fwd_top_data_memory;
    shared_ptr<primitive> fwd_bottom_data_primitive;
    int32_t num_, width_, height_, channels_;

};

// =====  MKLDNNInnerProductLayer =======================================
class MKLDNNInnerProductLayer : public MKLDNNLayer , public InnerProductLayer  {
public:
    explicit MKLDNNInnerProductLayer(const LayerParameter& param);
    virtual ~MKLDNNInnerProductLayer();
protected:
    virtual void Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
    // Customized methods
    virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
    void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);
private:
    void InitInnerProductFwd(const vector<Blob*>& bottom, const vector<Blob*>& top);

    shared_ptr<MKLDNNData > fwd_bottom_data, fwd_top_data, fwd_weights_data, fwd_bias_data;
    shared_ptr<inner_product_forward::primitive_desc> ipFwd_pd;

    MKLDNNPrimitive ipFwd;
    shared_ptr<memory> fwd_top_data_memory;
    shared_ptr<primitive> fwd_bottom_data_primitive, fwd_weights_data_primitive, fwd_bias_data_primitive;
    int32_t w_, h_;
};

// ===== MKLDNNPoolingLayer =======================================
class MKLDNNPoolingLayer : public MKLDNNLayer, public Layer {
public:
    explicit MKLDNNPoolingLayer(const LayerParameter& param)
            : MKLDNNLayer(param), Layer(param)
            , fwd_bottom_data(), fwd_top_data()
            , poolingFwd_pd()
            , indices_pd()
            , indices_memory(), fwd_top_data_memory()
            , fwd_bottom_data_primitive()
            , num_(0), channels_(0), width_(0), height_(0), width_out_(0), height_out_(0)
            , kernel_w_(0), kernel_h_(0), stride_w_(0), stride_h_(0)
            , pad_t_(0),pad_b_(0), pad_l_(0), pad_r_(0)
            , global_pooling_(false)
            , force_exclude_padding_flag_(false)
            {
            }
    ~MKLDNNPoolingLayer() {}
protected:
    virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
    virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);

    virtual inline const char* type() const { return "Pooling"; }
    virtual inline int ExactNumBottomBlobs() const { return 1; }
    virtual inline int MinTopBlobs() const { return 1; }
    // MAX POOL layers can output an extra top blob for the mask;
    // others can only output the pooled inputs.
    virtual inline int MaxTopBlobs() const {
        return (this->layer_param_.pooling_param().pool() == PoolingParameter_PoolMethod_MAX) ? 2 : 1;
    }
protected:
    virtual void Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
    virtual void compute_output_shape(const vector<Blob*>& bottom, const vector<Blob*>& top);

private:
    void InitPoolingFwd(const vector<Blob*>& bottom, const vector<Blob*>& top);

    shared_ptr<MKLDNNData> fwd_bottom_data, fwd_top_data;
    shared_ptr<pooling_forward::primitive_desc> poolingFwd_pd;
    shared_ptr<memory::primitive_desc> indices_pd;
    shared_ptr<memory> indices_memory, fwd_top_data_memory;
    MKLDNNPrimitive poolingFwd;
    shared_ptr<primitive> fwd_bottom_data_primitive;
    int32_t num_, channels_, width_, height_, width_out_, height_out_;
    int32_t kernel_w_, kernel_h_, stride_w_, stride_h_;
    int32_t  pad_t_, pad_b_, pad_l_, pad_r_;
    BlobInt max_idx_;
    bool global_pooling_;
    bool force_exclude_padding_flag_;
};


// =====  MKLDNNBatchNormLayer =======================================
class MKLDNNBatchNormLayer : public MKLDNNLayer, public Layer {
public:
    explicit MKLDNNBatchNormLayer(const LayerParameter& param)
        : MKLDNNLayer(param), Layer(param)
        , fwd_top_data(), fwd_bottom_data()
        , BatchNormFwd_pd()
        , scaleshift_memory()
        , output_memory()
        , input_primitive()
        {
    }
    ~MKLDNNBatchNormLayer() {}

protected:
    virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
    virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);
    virtual inline const char* type() const { return "BatchNorm"; }
    virtual void Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
private:
    void InitBatchNorm(const vector<Blob*>& bottom, const vector<Blob*>& top);
    void InitBatchNormFwdPrimitive(int stats_batch_idx);
    shared_ptr<memory> GetStatsBatchMemory(
      shared_ptr<MKLDNNMemoryDescriptor > mkldnn_data, int idx);
    void InitStatsBatchVars(int batch_size);
    shared_ptr<MKLDNNData > fwd_top_data, fwd_bottom_data;
    shared_ptr<batch_normalization_forward::primitive_desc> BatchNormFwd_pd;
    vector<MKLDNNPrimitive > BatchNormFwd;
    vector<shared_ptr<memory> > mean_memory, variance_memory;

    shared_ptr<memory> scaleshift_memory;
    shared_ptr<memory> output_memory;

    vector<shared_ptr<memory> > input_stats, output_stats;

    shared_ptr<primitive> input_primitive;

    int32_t num_, width_, height_, channels_;
    real_t eps_, moving_average_fraction_;
    bool use_weight_bias_, bias_term_, use_global_stats_;
    int num_stats_batches_;
    int stats_batch_size_;
    shared_ptr<Blob > scaleshift_blob_;
    shared_ptr<Blob > scaleshift_acc_;
};

// =====  MKLDNNEltwiseLayer =======================================
class MKLDNNEltwiseLayer : public MKLDNNLayer , public Layer  {
public:
  explicit MKLDNNEltwiseLayer(const LayerParameter& param)
    : MKLDNNLayer(param), Layer(param)
    , fwd_top_data(), fwd_bottom_data()
    , eltwiseFwd_pd()
    , fwd_top_data_memory()
    , fwd_bottom_data_primitives_()
    , num_(0), width_(0), height_(0), channels_(0)
    , num_bottoms_(0)
  {
  }
  ~MKLDNNEltwiseLayer() {}

protected:
    virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
    virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);
    virtual inline const char* type() const { return "Eltwise"; }
    virtual inline int MinBottomBlobs() const { return 2; }
    virtual inline int ExactNumTopBlobs() const { return 1; }
    virtual void Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
private:
    void InitEltwiseFwd(const vector<Blob*>& bottom, const vector<Blob*>& top);
    shared_ptr<MKLDNNData > fwd_top_data;
    vector<shared_ptr<MKLDNNData > > fwd_bottom_data;
    shared_ptr<sum::primitive_desc> eltwiseFwd_pd;
    MKLDNNPrimitive eltwiseFwd;

    shared_ptr<memory> fwd_top_data_memory;
    vector<shared_ptr<primitive>> fwd_bottom_data_primitives_;
    vector<primitive::at> fwd_bottom_data_primitives_at_;

    EltwiseParameter_EltwiseOp op_;
    vector<real_t > coeffs_;
    BlobInt max_idx_;
    int32_t num_, width_, height_, channels_;
    int32_t num_bottoms_;
    bool stable_prod_grad_;

};


// ===== MKLDNNConcatLayer ======================================
class MKLDNNConcatLayer : public MKLDNNLayer , public Layer {
public:
    explicit MKLDNNConcatLayer(const LayerParameter& param)
            : MKLDNNLayer(param), Layer(param),
            concatFwd_pd(), fwd_output_memory(),
            fwd_top_data(), fwd_bottom_data(), split_dims() {
    }
protected:
    virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
    virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);
    virtual inline const char* type() const { return "Concat"; }
    virtual void Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
private:
    void InitConcatFwd(const vector<Blob*>& bottom, const vector<Blob*>& top);

    shared_ptr<concat::primitive_desc> concatFwd_pd;
    shared_ptr<memory> fwd_output_memory;
    vector<shared_ptr<primitive>> fwd_input_primitives_;
    vector<primitive::at> fwd_input_primitives_at_;
    MKLDNNPrimitive concatFwd;
    shared_ptr<MKLDNNData > fwd_top_data;
    vector<shared_ptr<MKLDNNData > > fwd_bottom_data;
    vector<MKLDNNPrimitive > reorders;
    vector<int> split_dims;
    bool in_place_;

    int32_t num_, width_, height_, channels_, num_concats_;
    int concat_dimension;
};

// ===== MKLDNNSplitLayer ======================================
class MKLDNNSplitLayer : public MKLDNNLayer , public Layer {
public:
    explicit MKLDNNSplitLayer(const LayerParameter& param)
            : MKLDNNLayer(param), Layer(param) {}
    ~MKLDNNSplitLayer();

protected:
    virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
    virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);
    virtual inline const char* type() const { return "Split"; }
    virtual void Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
private:
    void InitSplitFwd(const vector<Blob*>& bottom, const vector<Blob*>& top);

  private:
    std::vector<size_t> sizes_src_;
    std::vector<size_t> strides_src_;
};

}  // namespace caffe
#endif  // USE_CUDNN
#endif  // #ifndef CAFFE_MKLDNN_LAYERS_HPP_