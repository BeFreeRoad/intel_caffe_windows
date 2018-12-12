#ifndef CAFFE_NET_HPP_
#define CAFFE_NET_HPP_

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "../../src/proto/caffe.pb.h"

namespace caffe {

class Layer;
class NetParameter;

/**
 * @brief Connects Layer%s together into a directed acyclic graph (DAG)
 *        specified by a NetParameter.
 *
 * TODO(dox): more thorough description.
 */
class CAFFE_API Net {
 public:
  explicit Net(const string& param_file);
  explicit Net(NetParameter& param) {
    Init(param);
  }

  /// @brief Initialize a network with a NetParameter.
  void Init(NetParameter& param);

  /**
   * @brief Run Forward and return the result.
   *
   * @param reshape, if your input data shape changes, reshape is need.
   */
  void Forward(bool reshape=true);

  /**
   * @brief Reshape all layers from bottom to top.
   *
   * This is useful to propagate changes to layer sizes without running
   * a forward pass, e.g. to compute output feature size.
   */
  void Reshape();
  /**
   * @brief Remove or Replace layers that the user specified should be excluded to increase
   *        computational performance.
   */
  static void CompileNet(const NetParameter& param,
                                    NetParameter* param_compiled);
    /**
  * @brief This is rule that analyze layer if it is of type Scale and if that is the case
  *        and previous layer which serves as input layer to Scale Layer is MKLBatchNorm
  *        then scale layer can be dropped
  */
  static void CompilationRuleRemoveScale(const NetParameter& param,
                                    NetParameter* param_compiled);
     /**
   * @brief If find "Conv--BN--Scale" in current network, merge BN and Scale layer into Convolution
   * layers, this optimization only works in caffe TEST phase now.
   */
  static void GetBlobConsumers(
                  std::vector<const LayerParameter*>& consumer_blobs,
                  const string& blob_name_to_find,
                  const NetParameter& param,
                  int layer_id_to_start_traversing_from);
    /**
  * @brief This is rule that analyze layer if it is of type MKLDNNReLU and if that is the case
  *        and previous layer which serves as input layer to MKLDNNReLU Layer is MKLDNNConvolution
  *        then MKLDNNReLU layer can be dropped
  */
  static void CompilationRuleConvReluFusion(const NetParameter& param,
                                    NetParameter* param_compiled);
    /**
  * @brief This is rule that analyze layer if it is of type MKLDNNReLU and if that is the case
  *        and previous layer which serves as input layer to MKLDNNReLU Layer is MKLDNNBN
  *        then MKLDNNReLU layer can be dropped
  */
  static void CompilationRuleFuseBnRelu(const NetParameter& param,
                                    NetParameter* param_compiled);
  static void ParseNetInplaceStatus(
    std::map<string, int>& inplace_blob_name_to_index,
    std::map<string, int>& specified_layer_blob_name_to_index,
    vector<vector<string>>& specified_layer_input_blob_names,
    NetParameter* param, const string& specified_layer_type);
  static void GetNeedToCancelInplaceLayers(
    vector<vector<const LayerParameter*>>& layer_pairs,
    std::map<string, int>& specified_layer_blob_name_to_index,
    std::map<string, int>& inplace_blob_name_to_index,
    vector<string>& each_blob_list, const NetParameter& param);
    /**
  * @brief This is rule analyze if layer is of type MKLBatchNorm
  *        and is to perform in place computation
  *        if positive then make it doing out-ofplace computation
  */
  static void CompilationRuleBNInplace(const NetParameter& param,
                                      NetParameter* param_compiled);

   /**
  * @brief This is rule analyze for general sparse.
  */
  static void CompilationRuleSparse(const NetParameter& param,
                                       NetParameter* param_compiled);
    /**
  * @brief This is rule analyze for conv/elt/relu fusion.
  */
  static void CompilationRuleConvSumFusion(const NetParameter& param,
                                              NetParameter* param_compiled);
  // For an already initialized net, CopyTrainedLayersFrom() copies the already
  // trained layers from another net parameter instance.
  /**
   * @brief For an already initialized net, copies the pre-trained layers from
   *        another Net.
   */
  void CopyTrainedLayersFrom(const NetParameter& param_inp);
  void CopyTrainedLayersFrom(const string& trained_filename);
  /// @brief Writes the net to a proto.
  void ToProto(NetParameter* param) const;

  /// @brief returns the network name.
  const string& name() const { return name_; }
  /// @brief returns the blob names
  const vector<string>& blob_names() const { return blob_names_; }
  /// @biref return the param names
  const vector<string>& param_names() const { return param_display_names_; }
  /// @brief returns the blobs
  const vector<shared_ptr<Blob> >& blobs() const {
    return blobs_;
  }
  /// @brief all parameters
  const vector<shared_ptr<Blob> >& params() const {
    return params_;
  }
  /**
   * @brief returns the bottom vecs for each layer -- usually you won't
   *        need this unless you do per-layer checks such as gradients.
   */
  const vector<vector<Blob*> >& bottom_vecs() const {
    return bottom_vecs_;
  }
  /**
   * @brief returns the top vecs for each layer -- usually you won't
   *        need this unless you do per-layer checks such as gradients.
   */
  const vector<vector<Blob*> >& top_vecs() const {
    return top_vecs_;
  }
  /**
   * @brief returns the params in every layer with id in params
   */
  const vector<vector<int> >& param_id_vecs() const {
    return param_id_vecs_;
  }
  /// @brief returns the ids of the top blobs of layer i
  const vector<int> & top_ids(int i) const {
    CHECK_GE(i, 0) << "Invalid layer id";
    CHECK_LT(i, top_id_vecs_.size()) << "Invalid layer id";
    return top_id_vecs_[i];
  }
  /// @brief returns the ids of the bottom blobs of layer i
  const vector<int> & bottom_ids(int i) const {
    CHECK_GE(i, 0) << "Invalid layer id";
    CHECK_LT(i, bottom_id_vecs_.size()) << "Invalid layer id";
    return bottom_id_vecs_[i];
  }
  bool has_blob(const string& blob_name) const;
  const shared_ptr<Blob> blob_by_name(const string& blob_name) const;

  /// @brief mark extra output named blob
  void MarkOutputs(const std::vector<std::string>& outs);

 protected:
  // Helpers for Init.
  /// @brief Append a new top blob to the net.
  void AppendTop(const NetParameter& param, const int layer_id,
                 const int top_id, std::set<string>* available_blobs,
                 std::map<string, int>* blob_name_to_idx);
  /// @brief Append a new bottom blob to the net.
  int AppendBottom(const NetParameter& param, const int layer_id,
                   const int bottom_id, std::set<string>* available_blobs,
                   std::map<string, int>* blob_name_to_idx);
  /// @brief Append a new parameter blob to the net.
  void AppendParam(const NetParameter& param, const int layer_id,
                   const int param_id);

  void PlaceMemory();

  /// @brief The network name
  string name_;
  /// @brief Individual layers in the net
  vector<shared_ptr<Layer> > layers_;
  vector<string> layer_names_;
  std::map<string, int> layer_names_index_;
  /// @brief the blobs storing intermediate results between the layer.
  vector<shared_ptr<Blob> > blobs_;
  vector<string> blob_names_;
  vector<int> blob_life_time_;
  std::map<string, int> blob_names_index_;
  /// @brief parameters in the network.
  vector<shared_ptr<Blob> > params_;
  vector<string> param_display_names_;
  vector<vector<int> > param_id_vecs_;
  std::map<string, int> param_names_index_;
  /// bottom_vecs stores the vectors containing the input for each layer.
  /// They don't actually host the blobs (blobs_ does), so we simply store
  /// pointers.
  vector<vector<Blob*> > bottom_vecs_;
  vector<vector<int> > bottom_id_vecs_;
  /// top_vecs stores the vectors containing the output for each layer
  vector<vector<Blob*> > top_vecs_;
  vector<vector<int> > top_id_vecs_;
  /// @brief The engine name
  string engine_name_;
  bool bn_scale_remove_;
  bool bn_scale_merge_;
  vector<string> kept_bn_layers_;
  DISABLE_COPY_AND_ASSIGN(Net);
};

/// @brief Read text net parameter, like xxx.prototxt
CAFFE_API shared_ptr<NetParameter> ReadTextNetParameterFromFile(const string& file);
CAFFE_API shared_ptr<NetParameter> ReadTextNetParameterFromBuffer(const char* buffer, int buffer_len);
/// @brief Read binary net parameter, like xxx.binaryproto
CAFFE_API shared_ptr<NetParameter> ReadBinaryNetParameterFromFile(const string& file);
CAFFE_API shared_ptr<NetParameter> ReadBinaryNetParameterFromBuffer(const char* buffer, int buffer_len);

}  // namespace caffe

#endif  // CAFFE_NET_HPP_
