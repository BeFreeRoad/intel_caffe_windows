#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/coded_stream.h>

#include "caffe/net.hpp"
#include "caffe/profiler.hpp"
#include "./layer.hpp"
#include "./util/math_functions.hpp"
#include "./util/upgrade_proto.hpp"
#include "./proto/caffe.pb.h"
#include "./util/remove_batch_norm.hpp"
#include "util/insert_splits.hpp"

namespace caffe {

Net::Net(const string& param_file) {
  NetParameter param;
  ReadNetParamsFromTextFileOrDie(param_file, &param);
  Init(param);
}
//mkldnn
void Net::GetBlobConsumers(
                  std::vector<const LayerParameter*>& consumer_blobs,
                  const string& blob_name_to_find,
                  const NetParameter& param,
                  int layer_id_to_start_traversing_from) {
  consumer_blobs.clear();
  // Validate values of ids of layers are <1..num_layers-1>
  CHECK_GE(layer_id_to_start_traversing_from, 1);
  CHECK_LT(layer_id_to_start_traversing_from, param.layer_size());

  // Traverse through layers to search the layer that consumes blob_name_to_find
  for (int i = layer_id_to_start_traversing_from; i < param.layer_size(); ++i) {
    // check bottom blobs if any of them is consuming given blob
    for (int j = 0; j < param.layer(i).bottom_size(); ++j) {
      if (param.layer(i).bottom(j).compare(blob_name_to_find) == 0) {
        consumer_blobs.push_back(&param.layer(i));
      }
    }
  }
}
void Net::CompilationRuleRemoveScale(const NetParameter& param,
                                    NetParameter* param_compiled) {
  bool merge_bn_scale = false;
  std::set<std::string> layers_to_drop;
  for (int i = 0; i < param.layer_size(); ++i) {
    LayerParameter* layer_param =
          (const_cast<NetParameter&>(param)).mutable_layer(i);
    bool layer_included = true;

    // Optimization rule 1:
    // - If we are having engine MKL2017 and Scale layer within a model
    // and input bottom comes from  BatchNorm of engine MKL2017
    // then we can remove Scale layer
    // and rename BatchNorm top blob after deleted Scale's top

    // Extension of optimization rule 1:
    // - If we are having engine MKLDNN and Scale layer within a model
    // and input bottom comes from  BatchNorm of engine MKLDNN
    // then we can remove Scale layer
    // and rename BatchNorm top blob after deleted Scale's top

        // If current layer is BatchNorm of MKL2017 engine..
    if (((layer_param->type().compare("BatchNorm") == 0) &&
         ((layer_param->batch_norm_param().engine() == BatchNormParameter_Engine_MKL2017) ||
          ((layer_param->batch_norm_param().engine() == BatchNormParameter_Engine_DEFAULT) &&
           (param.engine().compare("MKL2017") == 0)) ||
          (layer_param->engine().compare("MKL2017") == 0))) ||
        // If current layer is BatchNorm of MKLDNN engine..
        ((layer_param->type().compare("BatchNorm") == 0) &&
         ((layer_param->batch_norm_param().engine() == BatchNormParameter_Engine_MKLDNN) ||
          ((layer_param->batch_norm_param().engine() == BatchNormParameter_Engine_DEFAULT) &&
           (param.engine().compare("MKLDNN") == 0)) ||
          (layer_param->engine().compare("MKLDNN") == 0)))) {
      std::vector<const LayerParameter*> consumer_layer_params;
      GetBlobConsumers(consumer_layer_params,
                       layer_param->top(0),
                       param,
                       i+1 < param.layer_size() ? i+1 : i);
      const LayerParameter& consumer_layer_param =
                                    consumer_layer_params.size() > 0 ?
                                    *(consumer_layer_params[0]) : *layer_param;
      // Consumer layer of blob produced by BN
      // has to be Scale layer with one Input Blob
      if ((consumer_layer_param.type().compare("Scale") == 0) &&
           (consumer_layer_param.bottom_size() == 1)) {
        string& batchnorm_top_blob_name =
            const_cast<string&>(layer_param->top(0));
        const string& scale_top_blob_name = consumer_layer_param.top(0);
        // Mark Consumer layer (its name) as the one marked for dropping
        layers_to_drop.insert(consumer_layer_param.name());
        if (!merge_bn_scale) merge_bn_scale = true;

        // Replace BatchNorm top name with Scale top name
        batchnorm_top_blob_name.resize(scale_top_blob_name.size());
        batchnorm_top_blob_name.replace(0,
                                        scale_top_blob_name.size(),
                                        scale_top_blob_name);
        // Read the bias_term param of Scale Layer and set bias_term param
        // of MKLBatchNorm accordingly
        bool scale_bias_term = consumer_layer_param.
                               scale_param().bias_term();
        layer_param->mutable_batch_norm_param()->
        set_bias_term(scale_bias_term);
        if (consumer_layer_param.blobs_size() == 2) {
          layer_param->add_blobs()->CopyFrom(consumer_layer_param.blobs(0));
          layer_param->add_blobs()->CopyFrom(consumer_layer_param.blobs(1));
        }
        if (consumer_layer_param.param_size() == 2) {
          layer_param->add_param()->CopyFrom(consumer_layer_param.param(0));
          layer_param->add_param()->CopyFrom(consumer_layer_param.param(1));
        }
      }
    }

    if (layers_to_drop.find(layer_param->name()) != layers_to_drop.end()) {
      layer_included = false;
      LOG(INFO) << "Dropped layer: "
             << layer_param->name() << std::endl;
      // Remove dropped layer from the list of layers to be dropped
      layers_to_drop.erase(layers_to_drop.find(layer_param->name()));
    }

    if (layer_included) {
      param_compiled->add_layer()->CopyFrom(*layer_param);
    }
  }
  param_compiled->mutable_compile_net_state()->set_bn_scale_merge(merge_bn_scale);
}

void Net::CompilationRuleConvReluFusion(const NetParameter& param,
                                    NetParameter* param_compiled) {
  std::set<std::string> layers_to_drop;
  bool use_negative_slope = false;
  for (int i = 0; i < param.layer_size(); ++i) {
    LayerParameter* layer_param =
          (const_cast<NetParameter&>(param)).mutable_layer(i);
    bool layer_included = true;

    // Optimization rule 2:
    // - If we are having engine MKLDNN and ReLU layer within a model
    // and input bottom comes from  Convolution of engine MKLDNN
    // then we can remove ReLU layer
    // and rename Convolution top blob after deleted ReLU's top
    // Note: Currently merging of convolution and relu layers is feasible
    // If current layer is Convolution of MKLDNN engine..
    if ((layer_param->type().compare("Convolution") == 0) &&
        ((layer_param->convolution_param().engine() == ConvolutionParameter_Engine_MKLDNN) ||
         ((layer_param->convolution_param().engine() == ConvolutionParameter_Engine_DEFAULT) &&
          (layer_param->engine().compare(0, 6, "MKLDNN") == 0)) ||
         ((layer_param->convolution_param().engine() == ConvolutionParameter_Engine_DEFAULT) &&
          (layer_param->engine() == "") &&
          param.engine().compare(0, 6, "MKLDNN") == 0))) {
      std::vector<const LayerParameter*> consumer_layer_params;
      GetBlobConsumers(consumer_layer_params, layer_param->top(0),
                       param, i+1 < param.layer_size() ? i+1 : i);
      const LayerParameter& consumer_layer_param =
                                    consumer_layer_params.size() > 0 ?
                                    *(consumer_layer_params[0]) : *layer_param;

      // Consumer layer of blob produced by Conv
      // has to be ReLU layer with one Input Blob
      if ((consumer_layer_param.type().compare("ReLU") == 0) &&
          ((consumer_layer_param.relu_param().engine() == ReLUParameter_Engine_MKLDNN) ||
           ((consumer_layer_param.relu_param().engine() == ReLUParameter_Engine_DEFAULT) &&
            (consumer_layer_param.engine().compare(0, 6, "MKLDNN") == 0)) ||
           ((consumer_layer_param.relu_param().engine() == ReLUParameter_Engine_DEFAULT) &&
            (consumer_layer_param.engine() == "") &&
            (param.engine().compare(0, 6, "MKLDNN") == 0)))) {
        string& convolution_top_blob_name =
            const_cast<string&>(layer_param->top(0));

        float negative_slope1 =
                  consumer_layer_param.relu_param().negative_slope();
        if (negative_slope1 != 0) {
            use_negative_slope = true;
        }
        if(!negative_slope1) {
          const string& scale_top_blob_name = consumer_layer_param.top(0);
          // Mark Consumer layer (its name) as the one marked for dropping
          layers_to_drop.insert(consumer_layer_param.name());

          // Replace Convolution top name with ReLU top name
          convolution_top_blob_name.resize(scale_top_blob_name.size());
          convolution_top_blob_name.replace(0,
                                          scale_top_blob_name.size(),
                                          scale_top_blob_name);
        }
        if(!use_negative_slope) {
          layer_param->mutable_convolution_param()->set_relu(true);
          layer_param->mutable_convolution_param()->set_negative_slope(0);
        }
      }
    }

    if(!use_negative_slope) {
      if (layers_to_drop.find(layer_param->name()) != layers_to_drop.end()) {
        layer_included = false;
        LOG(INFO) << "Dropped layer: " << layer_param->name() << std::endl;
        // Remove dropped layer from the list of layers to be dropped
        layers_to_drop.erase(layers_to_drop.find(layer_param->name()));
      }
    }

    if (layer_included) {
      param_compiled->add_layer()->CopyFrom(*layer_param);
    }
  }
}

void Net::CompilationRuleFuseBnRelu(const NetParameter& param,
                                    NetParameter* param_compiled) {
  std::set<std::string> layers_to_drop;
  for (int i = 0; i < param.layer_size(); ++i) {
    LayerParameter* layer_param =
          (const_cast<NetParameter&>(param)).mutable_layer(i);
    bool layer_included = true;

    // Optimization rule BnRelu:
    // - If we are having engine MKLDNN and Relu layer within a model
    // and input bottom comes from  BatchNorm of engine MKLDNN
    // then we can remove Relu layer
    // and rename BatchNorm top blob after deleted Relu's top
    // If current layer is BatchNorm of MKLDNN engine..
    if (((layer_param->type().compare("BatchNorm") == 0) &&
         ((layer_param->batch_norm_param().engine() == BatchNormParameter_Engine_MKLDNN) ||
          ((layer_param->batch_norm_param().engine() == BatchNormParameter_Engine_DEFAULT) &&
           (layer_param->has_engine() == false)  &&
           (param.engine().compare("MKLDNN") == 0)) ||
          (param.engine() == "" && layer_param->engine().compare("MKLDNN") == 0)))) {
      std::vector<const LayerParameter*> consumer_layer_params;
      GetBlobConsumers(consumer_layer_params,
                       layer_param->top(0),
                       param,
                       i+1 < param.layer_size() ? i+1 : i);
      const LayerParameter& consumer_layer_param =
                                    consumer_layer_params.size() > 0 ?
                                    *(consumer_layer_params[0]) : *layer_param;
      // Consumer layer of blob produced by BN
      // has to be Relu layer with one Input Blob

      if ((consumer_layer_param.type().compare("ReLU") == 0) &&
          ((consumer_layer_param.relu_param().engine() == ReLUParameter_Engine_MKLDNN) ||
           ((consumer_layer_param.relu_param().engine() == ReLUParameter_Engine_DEFAULT) &&
            (consumer_layer_param.engine().compare(0, 6, "MKLDNN") == 0)) ||
           ((consumer_layer_param.relu_param().engine() == ReLUParameter_Engine_DEFAULT) &&
            (consumer_layer_param.engine() == "") &&
            (param.engine().compare(0, 6, "MKLDNN") == 0))) &&
             !consumer_layer_param.relu_param().negative_slope()) {
             // negative_slope should be zero
        string& batchnorm_top_blob_name =
            const_cast<string&>(layer_param->top(0));

        const string& relu_top_blob_name = consumer_layer_param.top(0);
        // Mark Consumer layer (its name) as the one marked for dropping
        layers_to_drop.insert(consumer_layer_param.name());

        // Replace BatchNorm top name with ReLU top name
        batchnorm_top_blob_name.resize(relu_top_blob_name.size());
        batchnorm_top_blob_name.replace(0,
                                          relu_top_blob_name.size(),
                                          relu_top_blob_name);
        // set relu flag in BN
        layer_param->mutable_batch_norm_param()->set_relu(true);

      }
    }

    if (layers_to_drop.find(layer_param->name()) != layers_to_drop.end()) {
      layer_included = false;
      LOG(INFO) << "Dropped layer: " << layer_param->name() << std::endl;
      // Remove dropped layer from the list of layers to be dropped
      layers_to_drop.erase(layers_to_drop.find(layer_param->name()));
    }

    if (layer_included) {
      param_compiled->add_layer()->CopyFrom(*layer_param);
    }
  }
}
void Net::ParseNetInplaceStatus(
    std::map<string, int>& inplace_blob_name_to_index,
    std::map<string, int>& specified_layer_blob_name_to_index,
    vector<vector<string>>& specified_layer_input_blob_names,
    NetParameter* param, const string& specified_layer_type) {
  for (int layer_index = 0; layer_index < param->layer_size(); ++layer_index) {
    LayerParameter* layer_param =
        (const_cast<NetParameter&>(*param)).mutable_layer(layer_index);

    if (!specified_layer_type.empty() &&
        layer_param->type().compare(specified_layer_type) != 0 &&
        layer_param->bottom_size() == 1 && layer_param->top_size() == 1 &&
        layer_param->bottom(0) == layer_param->top(0)) {
      inplace_blob_name_to_index[layer_param->bottom(0)] = layer_index;
    }

    if (!specified_layer_type.empty() &&
        layer_param->type().compare(specified_layer_type) == 0) {
      vector<string> blob_names;
      for (unsigned int blob_index = 0; blob_index < layer_param->bottom_size();
           blob_index++) {
        specified_layer_blob_name_to_index[layer_param->bottom(blob_index)] =
            layer_index;
        blob_names.push_back(layer_param->bottom(blob_index));
      }
      specified_layer_input_blob_names.push_back(blob_names);
    }
  }
}
void Net::GetNeedToCancelInplaceLayers(
    vector<vector<const LayerParameter*>>& layer_pairs,
    std::map<string, int>& specified_layer_blob_name_to_index,
    std::map<string, int>& inplace_blob_name_to_index,
    vector<string>& each_blob_list, const NetParameter& param) {
  if (param.engine().compare("MKLDNN") != 0 || each_blob_list.size() == 1)
    return;

  layer_pairs.clear();

  vector<const LayerParameter*> each_layer_pair;

  each_blob_list.erase(each_blob_list.begin());

  for (auto blob_name : each_blob_list) {
    each_layer_pair.clear();
    if (inplace_blob_name_to_index.find(blob_name) ==
            inplace_blob_name_to_index.end() ||
        specified_layer_blob_name_to_index.find(blob_name) ==
            specified_layer_blob_name_to_index.end()) {
      continue;
    }

    LayerParameter* bottom_layer =
        (const_cast<NetParameter&>(param))
            .mutable_layer(inplace_blob_name_to_index[blob_name]);
    LayerParameter* top_layer =
        (const_cast<NetParameter&>(param))
            .mutable_layer(specified_layer_blob_name_to_index[blob_name]);
    each_layer_pair.push_back(bottom_layer);
    each_layer_pair.push_back(top_layer);

    layer_pairs.push_back(each_layer_pair);
  }
}
void Net::CompilationRuleBNInplace(const NetParameter& param,
                                      NetParameter* param_compiled) {
  for (int i = 0; i < param.layer_size(); ++i) {
    LayerParameter* layer_param =
        (const_cast<NetParameter&>(param)).mutable_layer(i);

    // Optimization rule 3:
    // - If we are having engine MKL2017 and Batch Normalization
    // doing inplace computation then
    // to improve performance we create another top buffer
    // and make other layers consuming BatchNorm top to use new buffer

    // If current layer is BatchNorm of MKL2017 engine..
    if (((layer_param->type().compare("BatchNorm") == 0) &&
         (layer_param->batch_norm_param().engine() ==
              BatchNormParameter_Engine_MKL2017 ||
          ((layer_param->batch_norm_param().engine() ==
            BatchNormParameter_Engine_DEFAULT) &&
           param.engine().compare("MKL2017") == 0))) &&
        (layer_param->top(0) == layer_param->bottom(0))) {
      std::string& batch_norm_top = const_cast<string&>(layer_param->top(0));
      std::vector<const LayerParameter*> consumer_layer_params;
      GetBlobConsumers(consumer_layer_params, batch_norm_top, param,
                       i + 1 < param.layer_size() ? i + 1 : i);

      for (std::vector<const LayerParameter*>::iterator it =
               consumer_layer_params.begin();
           it != consumer_layer_params.end(); ++it) {
        // If consumer is computing inplace then modify top as well
        if (((*it)->top_size() > 0) &&
            ((*it)->bottom(0).compare((*it)->top(0)) == 0)) {
          // Modify consumer top
          const_cast<string&>((*it)->top(0)).append("_x");
        }

        // Modify consumer bottom. Sometimes searched
        // buffer is under higher bottom index than 0 eg.
        // In case of Eltwise
        for (unsigned int i = 0; i < (*it)->bottom_size(); ++i) {
          if ((*it)->bottom(i).compare(batch_norm_top) == 0) {
            const_cast<string&>((*it)->bottom(i)).append("_x");
          }
        }
      }
      // Modify top so it is diffrent from bottom
      batch_norm_top.append("_x");
    }

    param_compiled->add_layer()->CopyFrom(*layer_param);
  }

  return;
}

void Net::CompilationRuleSparse(const NetParameter& param,
                                       NetParameter* param_compiled) {
  //TODO: Verify the convergence of the sparse model
  if (param.engine().compare("MKLDNN") != 0) {
    param_compiled->CopyFrom(param);
    return;
  }

  LayerParameter* potential_sparse_layer = NULL;
  LayerParameter* confirmed_sparse_layer = NULL;
  LayerParameter* layer_param = NULL;

  std::map<string, string> bottom_blob_layer_mapping;
  // top blob as the key and  its layer name as the value
  std::map<string, string> top_blob_layer_mapping;
  std::map<string, std::vector<LayerParameter*>> sparse_layer_name_mapping;  // key is layer name
  std::vector<LayerParameter*> trigger_sparse_layers;
  std::map<string, int> conv_layer_id_mapping;
  std::map<string, int> eltwise_layer_id_mapping;
  std::map<string, int> layer_name_id_mapping;

  std::map<int, int> pooling_layer_id_stride;  // saves the layer's id which
                                               // need to add pooling layer;
  std::map<int, int> conv_layer_id_stride;  // saves the conv layer's id which
                                            // need to modify its stride;
  std::map<int, string> pooling_layer_id_top_blob;

  // step 1 get topology details, such as layer name/id mapping
  for (int index = 0; index < param.layer_size(); index++) {
    layer_param = (const_cast<NetParameter&>(param)).mutable_layer(index);
    layer_name_id_mapping[layer_param->name()] = index;

    for (int j = 0; j < layer_param->top_size(); j++) {
      top_blob_layer_mapping[layer_param->top(j)] = layer_param->name();
    }

    for (int k = 0; k < layer_param->bottom_size(); k++) {
      bottom_blob_layer_mapping[layer_param->bottom(k)] = layer_param->name();
    }

    if (layer_param->type().compare("Eltwise") == 0) {
      eltwise_layer_id_mapping[layer_param->name()] = index;
    }

    if (layer_param->type().compare("Convolution") == 0 &&
        layer_param->has_convolution_param() &&
        layer_param->convolution_param().kernel_size_size() > 0 &&
        layer_param->convolution_param().stride_size() > 0) {
      conv_layer_id_mapping[layer_param->name()] = index;

      if (layer_param->convolution_param().kernel_size(0) > 1) {
        potential_sparse_layer = layer_param;
      } else if (layer_param->convolution_param().kernel_size(0) == 1 &&
                 layer_param->convolution_param().stride(0) > 1  &&
          (layer_param->convolution_param().pad_size() == 0 ||
           (layer_param->convolution_param().pad_size() > 0 &&
            layer_param->convolution_param().pad(0) == 0))) {
        if (potential_sparse_layer == NULL)
            continue;
        confirmed_sparse_layer = potential_sparse_layer;

        if (trigger_sparse_layers.size() > 0) {
          for (int j = 0; j < trigger_sparse_layers.size(); j++) {
            if (top_blob_layer_mapping[trigger_sparse_layers[j]->bottom(0)] !=
                top_blob_layer_mapping[layer_param->bottom(0)]) {
              trigger_sparse_layers.clear();
              break;
            }
          }
          trigger_sparse_layers.push_back(layer_param);

          sparse_layer_name_mapping[confirmed_sparse_layer->name()] =
              trigger_sparse_layers;
        } else {
          trigger_sparse_layers.push_back(layer_param);
        }
      }
    }
  }

  if(trigger_sparse_layers.size() > 1)
    sparse_layer_name_mapping[confirmed_sparse_layer->name()] = trigger_sparse_layers;

  std::map<string, std::vector<LayerParameter*>>::iterator sparse_it =
      sparse_layer_name_mapping.begin();
  while (sparse_it != sparse_layer_name_mapping.end() && sparse_it->second.size() > 1) {

    if (sparse_it->second[0]->convolution_param().stride(0) !=
        sparse_it->second[1]->convolution_param().stride(0)) {
          continue;
    }
    LayerParameter* sparse_layer_param =
        (const_cast<NetParameter&>(param))
            .mutable_layer(layer_name_id_mapping[sparse_it->first]);
    int updated_stride_value =
        sparse_layer_param->convolution_param().stride(0) *
        sparse_it->second[0]->convolution_param().stride(0);
    conv_layer_id_stride[conv_layer_id_mapping[sparse_it->first]] =
        updated_stride_value;
    conv_layer_id_stride[conv_layer_id_mapping[sparse_it->second[0]->name()]] = 1;
    conv_layer_id_stride[conv_layer_id_mapping[sparse_it->second[1]->name()]] = 1;

    std::map<string, int>::iterator eltwise_iter = eltwise_layer_id_mapping.begin();
    while (eltwise_iter != eltwise_layer_id_mapping.end()) {
      // it means there is a eltwise layer between the layer need to sparse and
      // the  layer triggers the sparse
      if (conv_layer_id_mapping[sparse_it->first] < eltwise_iter->second &&
          eltwise_iter->second < conv_layer_id_mapping[sparse_it->second[0]->name()]) {
        break;  // now eltwise_iter stands for eltwise layer
      }
      eltwise_iter++;
    }

    std::vector<int> need_add_pooling_layer_id;
    LayerParameter* eltwise_layer_param =
        (const_cast<NetParameter&>(param)).mutable_layer(eltwise_iter->second);
    for (int k = 0; k < eltwise_layer_param->bottom_size() - 1; k++) {
      need_add_pooling_layer_id.push_back(
          layer_name_id_mapping
              [top_blob_layer_mapping[eltwise_layer_param->bottom(k)]]);
      int pooling_layer_id = layer_name_id_mapping
          [top_blob_layer_mapping[eltwise_layer_param->bottom(k)]];
      pooling_layer_id_stride[pooling_layer_id] = updated_stride_value;
      pooling_layer_id_top_blob[pooling_layer_id] =
          eltwise_layer_param->bottom(k);
    }
    sparse_it++;
  }

  for (int i = 0; i < param.layer_size(); i++) {
    LayerParameter* each_layer_param =
        (const_cast<NetParameter&>(param)).mutable_layer(i);
    if (conv_layer_id_stride.find(i) != conv_layer_id_stride.end()) {
      each_layer_param->mutable_convolution_param()->set_stride(
          0, conv_layer_id_stride[i]);
    } else if (pooling_layer_id_stride.find(i) !=
               pooling_layer_id_stride.end()) {
      param_compiled->add_layer()->CopyFrom(*each_layer_param);
      each_layer_param = param_compiled->add_layer();
      each_layer_param->Clear();
      each_layer_param->set_type("Pooling");
      each_layer_param->set_name(pooling_layer_id_top_blob[i] + "_p");

      each_layer_param->add_bottom(pooling_layer_id_top_blob[i]);
      each_layer_param->add_top(pooling_layer_id_top_blob[i] + "_p");

      each_layer_param->mutable_pooling_param()->set_stride(
          pooling_layer_id_stride[i]);
      each_layer_param->mutable_pooling_param()->set_kernel_size(1);
      each_layer_param->mutable_pooling_param()->set_pool(
          PoolingParameter_PoolMethod_MAX);

      int target_layer_id = layer_name_id_mapping
          [bottom_blob_layer_mapping[pooling_layer_id_top_blob[i]]];
      LayerParameter* target_layer_param =
          (const_cast<NetParameter&>(param)).mutable_layer(target_layer_id);
      int target_blob_index = 0;
      bool found_blob_flag = false;
      for (; target_blob_index < target_layer_param->bottom_size();
           target_blob_index++) {
        if (target_layer_param->bottom(target_blob_index) ==
            pooling_layer_id_top_blob[i]) {
          found_blob_flag = true;
          break;
        }
      }
      if (found_blob_flag) {
        target_layer_param->set_bottom(target_blob_index,
                                       pooling_layer_id_top_blob[i] + "_p");
        continue;
      }
    }
    param_compiled->add_layer()->CopyFrom(*each_layer_param);
  }
}

void Net::CompilationRuleConvSumFusion(const NetParameter& param,
                                              NetParameter* param_compiled) {
  // only apply this rule for inference(TEST) phase
  if (param.engine().compare("MKLDNN") != 0) {
    param_compiled->CopyFrom(param);
    return;
  }
  string blob_need_to_insert;
  LayerParameter* need_to_convert_layer = NULL;
  bool has_relu_flag = true;
  bool need_fusion_flag;
  std::set<string> invalid_fusion_blob_names;


  for (int i = 0; i < param.layer_size(); i++) {
    need_fusion_flag = true;
    LayerParameter* layer_param =
        (const_cast<NetParameter&>(param)).mutable_layer(i);
    if (layer_param->type().compare("Split") == 0 &&
        layer_param->top_size() > 2) {
      for (int j = 0; j < layer_param->top_size() - 1; j++) {
        invalid_fusion_blob_names.insert(layer_param->top(j));
      }
    }
    if (layer_param->type().compare("Convolution") == 0 &&
        (layer_param->has_engine() == false ||
         (layer_param->has_engine() == true &&
          layer_param->engine().compare("MKLDNN") == 0))) {
      std::vector<const LayerParameter*> child_layers_params;
      GetBlobConsumers(child_layers_params, layer_param->top(0),
                                   param,
                                   i + 1 < param.layer_size() ? i + 1 : i);
      if (child_layers_params.size() > 0 &&
          child_layers_params[0]->type().compare("Eltwise") == 0) {

        for (int k = 0; k < child_layers_params[0]->bottom_size(); k++) {
          if (invalid_fusion_blob_names.count(
                  child_layers_params[0]->bottom(k)) > 0) {
            need_fusion_flag = false;
            break;
          }
        }
        if (!need_fusion_flag) {
          param_compiled->add_layer()->CopyFrom(*layer_param);
          continue;
        }

        std::vector<const LayerParameter*> grand_child_layers_params;

        GetBlobConsumers(grand_child_layers_params,
                                     child_layers_params[0]->top(0), param,
                                     i + 1 < param.layer_size() ? i + 1 : i);
        const LayerParameter& grand_child_layer_param =
            grand_child_layers_params.size() > 0
                ? *(grand_child_layers_params[0])
                : *layer_param;

        if (grand_child_layer_param.type().compare("ReLU") != 0) {
          has_relu_flag = false;
        }

        if (child_layers_params[0]->bottom(0) == layer_param->top(0)) {
          param_compiled->add_layer()->CopyFrom(*layer_param);
          need_to_convert_layer = layer_param;
          continue;
        }

        if (has_relu_flag) {
          const_cast<string&>(layer_param->top(0)) =
              grand_child_layer_param.top(0);
        } else {
          const_cast<string&>(layer_param->top(0)) =
              child_layers_params[0]->top(0);
        }

        if (need_to_convert_layer != NULL) {
          layer_param->add_bottom(
              const_cast<string&>(need_to_convert_layer->top(0)));
          need_to_convert_layer = NULL;
        } else {
          layer_param->add_bottom(
              const_cast<string&>(child_layers_params[0]->bottom(0)));
        }

        if (has_relu_flag) {
          i += 2;  // skip next eltwise and relu
          layer_param->mutable_convolution_param()->set_relu(true);
        } else {
          i += 1;
        }

        layer_param->mutable_convolution_param()->set_fusion_type(ConvolutionParameter::SUM_FUSION);
        size_t coeff_size = child_layers_params[0]->eltwise_param().coeff_size();
        if (coeff_size > 0)
        {
          for (int i = 0; i < coeff_size; ++i)
          {
//            layer_param->mutable_convolution_param()->add_coeff(child_layers_params[0]->eltwise_param().coeff(i));
          }
        }
      }
    }

    param_compiled->add_layer()->CopyFrom(*layer_param);
  }

  return;
}
void Net::CompileNet(const NetParameter& param,
    NetParameter* param_compiled) {

  #define NUM_OF_RULES sizeof(CompileRules)/sizeof(CompileRules[0])
  #define COMPILE_BN_FOLDING_INDEX 0
  #define COMPILE_CONV_RELU_FUSION_INDEX 2
  #define COMPILE_BN_RELU_FUSION_INDEX 3
  #define COMPILE_SPARSE_INDEX 5
  #define COMPILE_CONV_SUM_FUSION_INDEX 6
  int i, current = 0;
  NetParameter param_temp[2];
  void (*CompileRules[]) (const NetParameter& param, NetParameter* param_compiled) =
    {RemoveBNScale,CompilationRuleRemoveScale, CompilationRuleConvReluFusion,
    CompilationRuleFuseBnRelu, CompilationRuleBNInplace, CompilationRuleSparse,
    CompilationRuleConvSumFusion};

  bool disabled[NUM_OF_RULES] = {true};

#ifdef DISABLE_BN_FOLDING
  disabled[COMPILE_BN_FOLDING_INDEX] = true;
#endif
#ifdef DISABLE_CONV_RELU_FUSION
  disabled[COMPILE_CONV_RELU_FUSION_INDEX] = true;
#endif
#ifdef DISABLE_BN_RELU_FUSION
  disabled[COMPILE_BN_RELU_FUSION_INDEX] = true;
#endif
#ifdef DISABLE_CONV_SUM_FUSION
  disabled[COMPILE_CONV_SUM_FUSION_INDEX] = true;
#endif
#ifdef DISABLE_SPARSE
  disabled[COMPILE_SPARSE_INDEX] = true;
#endif

  param_temp[current].CopyFrom(param);
  for (i = 0; i < NUM_OF_RULES; i++)
    if (!disabled[i]) {
      param_temp[1 - current].CopyFrom(param_temp[current]);
      param_temp[1 - current].clear_layer();   // Remove layers
      (*CompileRules[i]) (param_temp[current], &param_temp[1 - current]);
      current = 1 - current;
    }
  param_compiled->CopyFrom(param_temp[current]);
  #undef NUM_OF_RULES
  #undef COMPILE_BN_FOLDING_INDEX
  #undef COMPILE_CONV_RELU_FUSION_INDEX
  #undef COMPILE_BN_RELU_FUSION_INDEX
  #undef COMPILE_SPARSE_INDEX
  #undef COMPILE_CONV_SUM_FUSION_INDEX
}


void Net::Init(NetParameter& param) {
  //mkldnn
#ifdef USE_MKLDNN
  static bool executed = false;
  if (param.engine() == ""){
    param.set_engine("MKLDNN");
  }

  engine_name_ = param.engine();
  NetParameter compiled_param;
    // Transform Net (merge layers etc.) improve computational performance
  CompileNet(param, &compiled_param);
  param = compiled_param;
  this->bn_scale_remove_ = param.compile_net_state().bn_scale_remove();
  this->bn_scale_merge_ = param.compile_net_state().bn_scale_merge();
  int kept_bn_layers_num = param.compile_net_state().kept_bn_layers_size();
  for (int idx = 0; idx < kept_bn_layers_num; ++idx) {
    this->kept_bn_layers_.push_back(param.compile_net_state().kept_bn_layers(idx));
  }
#endif

    std::cout << param.DebugString();
  // Basically, build all the layers and set up their connections.
  name_ = param.name();
  std::map<string, int> blob_name_to_idx;
  std::set<string> available_blobs;
  // For each layer, set up its input and output
  bottom_vecs_.resize(param.layer_size());
  top_vecs_.resize(param.layer_size());
  bottom_id_vecs_.resize(param.layer_size());
  top_id_vecs_.resize(param.layer_size());
  param_id_vecs_.resize(param.layer_size());
  for (int layer_id = 0; layer_id < param.layer_size(); ++layer_id) {
    // Setup layer.
    const LayerParameter& layer_param = param.layer(layer_id);
    layers_.push_back(LayerRegistry::CreateLayer(layer_param));
    layer_names_.push_back(layer_param.name());
    // Figure out this layer's input and output
    const int num_bottom = layer_param.bottom_size();
    for (int bottom_id = 0; bottom_id < num_bottom; ++bottom_id) {
      AppendBottom(param, layer_id, bottom_id, &available_blobs, &blob_name_to_idx);
    }
    const int num_top = layer_param.top_size();
    for (int top_id = 0; top_id < num_top; ++top_id) {
      AppendTop(param, layer_id, top_id, &available_blobs, &blob_name_to_idx);
    }
    // After this layer is connected, set it up.
    layers_[layer_id]->SetUp(bottom_vecs_[layer_id], top_vecs_[layer_id]);
    // Layer Parameters
    const int num_param_blobs = layers_[layer_id]->blobs().size();
    for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
      AppendParam(param, layer_id, param_id);
    }
  }
  CHECK_EQ(std::string(layers_[0]->type()), std::string("Input"))
      << "Network\'s first layer should be Input Layer.";
  // for most case, not fully convolutional network, hold input data will be convenient
  for (int blob_id : top_id_vecs_[0]) {
    blob_life_time_[blob_id] = layers_.size();
  }
  for (size_t blob_id = 0; blob_id < blob_names_.size(); ++blob_id) {
    blob_names_index_[blob_names_[blob_id]] = blob_id;
    blobs_[blob_id]->set_name(blob_names_[blob_id]);
  }
  for (size_t layer_id = 0; layer_id < layer_names_.size(); ++layer_id) {
    layer_names_index_[layer_names_[layer_id]] = layer_id;
  }
}

// Helper for Net::Init: add a new top blob to the net.
void Net::AppendTop(const NetParameter& param, const int layer_id,
                    const int top_id, std::set<string>* available_blobs,
                    std::map<string, int>* blob_name_to_idx) {
  shared_ptr<LayerParameter> layer_param(
      new LayerParameter(param.layer(layer_id)));
  const string& blob_name = (layer_param->top_size() > top_id) ?
      layer_param->top(top_id) : "(automatic)";
  // Check if we are doing in-place computation
  if (blob_name_to_idx && layer_param->bottom_size() > top_id &&
      blob_name == layer_param->bottom(top_id)) {
    // In-place computation
    int blob_id = (*blob_name_to_idx)[blob_name];
    top_vecs_[layer_id].push_back(blobs_[blob_id].get());
    top_id_vecs_[layer_id].push_back(blob_id);
    blob_life_time_[blob_id] = std::max(blob_life_time_[blob_id], layer_id + 1);
  } else if (blob_name_to_idx &&
             blob_name_to_idx->find(blob_name) != blob_name_to_idx->end()) {
    // If we are not doing in-place computation but have duplicated blobs,
    // raise an error.
    LOG(FATAL) << "Top blob '" << blob_name
               << "' produced by multiple sources.";
  } else {
    // Normal output.
    shared_ptr<Blob> blob_pointer(new Blob);
    const int blob_id = blobs_.size();
    blobs_.push_back(blob_pointer);
    blob_names_.push_back(blob_name);
    blob_life_time_.push_back(layer_id + 1);
    if (blob_name_to_idx) { (*blob_name_to_idx)[blob_name] = blob_id; }
    top_id_vecs_[layer_id].push_back(blob_id);
    top_vecs_[layer_id].push_back(blob_pointer.get());
  }
  if (available_blobs) { available_blobs->insert(blob_name); }
}

// Helper for Net::Init: add a new bottom blob to the net.
int Net::AppendBottom(const NetParameter& param, const int layer_id,
                      const int bottom_id, std::set<string>* available_blobs,
                      std::map<string, int>* blob_name_to_idx) {
  const LayerParameter& layer_param = param.layer(layer_id);
  const string& blob_name = layer_param.bottom(bottom_id);
  if (available_blobs->find(blob_name) == available_blobs->end()) {
    LOG(FATAL) << "Unknown bottom blob '" << blob_name << "' (layer '"
               << layer_param.name() << "', bottom index " << bottom_id << ")";
  }
  const int blob_id = (*blob_name_to_idx)[blob_name];
  bottom_vecs_[layer_id].push_back(blobs_[blob_id].get());
  bottom_id_vecs_[layer_id].push_back(blob_id);
  blob_life_time_[blob_id] = std::max(blob_life_time_[blob_id], layer_id);
  return blob_id;
}

void Net::AppendParam(const NetParameter& param, const int layer_id,
                      const int param_id) {
  const LayerParameter& layer_param = layers_[layer_id]->layer_param();
  const int param_size = layer_param.param_size();
  string param_name =
    (param_size > param_id) ? layer_param.param(param_id).name() : "";
  if (param_name.size()) {
    param_display_names_.push_back(param_name);
  }
  else {
    std::ostringstream param_display_name;
    param_display_name << layer_param.name() << "_" << param_id;
    param_display_names_.push_back(param_display_name.str());
  }
  const int net_param_id = params_.size();
  params_.push_back(layers_[layer_id]->blobs()[param_id]);
  param_id_vecs_[layer_id].push_back(net_param_id);
}

void Net::PlaceMemory() {
  // get shape info
//  this->Reshape();
//  // place
//  using BlobPair = std::pair<size_t, Blob*>;
//  std::multimap<size_t, Blob*> pool;
//  for (int i = 0; i < layers_.size(); ++i) {
//    // blobs used by layer i
//    DLOG(INFO) << "[MemPlace] Layer " << layer_names_[i];
//    std::vector<Blob*> temps = layers_[i]->GetTempBlobs();
//    std::vector<Blob*>& bottoms = bottom_vecs_[i];
//    std::vector<Blob*>& tops = top_vecs_[i];
//    // blobs need to place memory
//    std::vector<BlobPair> blobs;
//    blobs.reserve(temps.size() + tops.size());
//    for (auto* blob : temps) {
//      blob->ResetMemory();
//      blobs.push_back(std::make_pair(blob->count(), blob));
//    }
//    for (auto* blob : tops) {
//      bool should_place = true;
//      // check inplace
//      for (auto* bottom_blob : bottoms) {
//        if (bottom_blob == blob) {
//          should_place = false;
//          break;
//        }
//      }
//      if (should_place) {
//        blob->ResetMemory();
//        blobs.push_back(std::make_pair(blob->count(), blob));
//      }
//    }
//    std::sort(blobs.begin(), blobs.end(), [](const BlobPair& x, const BlobPair& y) {
//      return x.first > y.first;
//    });
//    // search pool to place memory if possible
//    for (auto& p : blobs) {
//      size_t size = p.first;
//      Blob* blob = p.second;
//      auto it = pool.lower_bound(size);
//      if (it != pool.end() && it->first <= size * 2) {
//        DLOG(INFO) << "[MemPlace] Share " << blob->name() << "(" << size << ") with " << it->second->name() << "(" << it->first << ")";
//        Blob* share = it->second;
//        blob->ShareData(*share);
//        pool.erase(it);
//      }
//      else {
//        DLOG(INFO) << "[MemPlace] Alloc " << blob->name() << "(" << size << ")";
//      }
//    }
//    // put unused blob to pool
//    for (int blob_idx : bottom_id_vecs_[i]) {
//      if (blob_life_time_[blob_idx] <= i) {
//        DLOG(INFO) << "[MemPlace] Put " << blobs_[blob_idx]->name() << "(" << blobs_[blob_idx]->capacity() << ") to Pool";
//        pool.insert(std::make_pair(blobs_[blob_idx]->capacity(), blobs_[blob_idx].get()));
//      }
//    }
//    for (auto* blob : temps) {
//      DLOG(INFO) << "[MemPlace] Put " << blob->name() << "(" << blob->capacity() << ") to Pool";
//      pool.insert(std::make_pair(blob->capacity(), blob));
//    }
//  }
}

void Net::Forward(bool reshape) {
  // static place memory
  if (reshape) {
//    PlaceMemory();
  }
  // forward network
  Profiler *profiler = Profiler::Get();
  for (int i = 0; i < layers_.size(); ++i) {
    // LOG(INFO) << "Forwarding " << layer_names_[i];
    profiler->ScopeStart(layer_names_[i].c_str());
    layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);
    profiler->ScopeEnd();
  }
  // sync gpu data
  if (Caffe::mode() == Caffe::GPU) {
    profiler->ScopeStart("Sync");
    for (auto* blob : top_vecs_[layers_.size() - 1]) {
      blob->cpu_data();
    }
    profiler->ScopeEnd();
  }
}

void Net::Reshape() {
  for (int i = 0; i < layers_.size(); ++i) {
    layers_[i]->Reshape(bottom_vecs_[i], top_vecs_[i]);
  }
}

void Net::CopyTrainedLayersFrom(const NetParameter& param_inp) {
#ifdef USE_MKLDNN
  NetParameter param_tmp = param_inp;
  NetParameter &param = param_tmp;
  param.set_engine(engine_name_);
  param_tmp.mutable_state()->set_phase(TEST);
  param_tmp.mutable_compile_net_state()->set_is_init(false);
  for (vector<string>::iterator it = this->kept_bn_layers_.begin(); it != this->kept_bn_layers_.end(); it++) {
    param_tmp.mutable_compile_net_state()->add_kept_bn_layers(*it);
  }
  int num_source_layers = param.layer_size();
  for (int i = 0; i < num_source_layers; ++i) {
    LayerParameter* source_layer = param.mutable_layer(i);
    const string& source_layer_name = source_layer->name();
    int target_layer_id = 0;
    while (target_layer_id != layer_names_.size() &&
        layer_names_[target_layer_id] != source_layer_name) {
      ++target_layer_id;
    }
    if (target_layer_id == layer_names_.size()) {
      continue;
    }
    const LayerParameter& layer_param = layers_[target_layer_id]->layer_param();
    const string& engine_name = layer_param.engine();
    source_layer->set_engine(engine_name);
    if ((layer_param.type().compare("BatchNorm") == 0) &&
        (layer_param.batch_norm_param().has_engine())) {
      source_layer->mutable_batch_norm_param()->set_engine(layer_param.batch_norm_param().engine());
    }
  }
  NetParameter param_compiled;
  CompileNet(param, &param_compiled);
  param = param_compiled;
  num_source_layers = param.layer_size();
  for (int i = 0; i < num_source_layers; ++i) {
    const LayerParameter& source_layer = param.layer(i);
    const string& source_layer_name = source_layer.name();
    int target_layer_id = 0;
    while (target_layer_id != layer_names_.size() &&
        layer_names_[target_layer_id] != source_layer_name) {
      ++target_layer_id;
    }
    if (target_layer_id == layer_names_.size()) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob > >& target_blobs =
        layers_[target_layer_id]->blobs();
#else
  int num_source_layers = param_inp.layer_size();
  for (int i = 0; i < num_source_layers; ++i) {
    const LayerParameter& source_layer = param_inp.layer(i);
    const string& source_layer_name = source_layer.name();
    int target_layer_id = 0;
    while (target_layer_id != layer_names_.size() &&
           layer_names_[target_layer_id] != source_layer_name) {
      ++target_layer_id;
    }
    if (target_layer_id == layer_names_.size()) {
      continue;
    }
    vector<shared_ptr<Blob> >& target_blobs =
        layers_[target_layer_id]->blobs();
#endif
    CHECK_EQ(target_blobs.size(), source_layer.blobs_size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      if (!target_blobs[j]->ShapeEquals(source_layer.blobs(j))) {
        Blob source_blob;
        const bool kReshape = true;
        source_blob.FromProto(source_layer.blobs(j), kReshape);
        LOG(FATAL) << "Cannot copy param " << j << " weights from layer '"
            << source_layer_name << "'; shape mismatch.  Source param shape is "
            << source_blob.shape_string() << "; target param shape is "
            << target_blobs[j]->shape_string() << ". "
            << "To learn this layer's parameters from scratch rather than "
            << "copying from a saved net, rename the layer.";
      }
      const bool kReshape = false;
      target_blobs[j]->FromProto(source_layer.blobs(j), kReshape);
    }
  }
}

void Net::MarkOutputs(const std::vector<std::string>& outs) {
  for (auto& name : outs) {
    auto it = blob_names_index_.find(name);
    if (it == blob_names_index_.end()) {
      LOG(FATAL) << "blob (" << name << ") is not availiable in Net";
    }
    int blob_id = it->second;
    blob_life_time_[blob_id] = layers_.size();
  }
}

void Net::CopyTrainedLayersFrom(const string& trained_filename) {
  NetParameter param;
  ReadNetParamsFromBinaryFileOrDie(trained_filename, &param);
  CopyTrainedLayersFrom(param);
}

void Net::ToProto(NetParameter* param) const {
  param->Clear();
  param->set_name(name_);
  // Add bottom and top
  for (int i = 0; i < layers_.size(); ++i) {
    LayerParameter* layer_param = param->add_layer();
    layers_[i]->ToProto(layer_param);
  }
}

bool Net::has_blob(const string& blob_name) const {
  return blob_names_index_.find(blob_name) != blob_names_index_.end();
}

const shared_ptr<Blob> Net::blob_by_name(const string& blob_name) const {
  shared_ptr<Blob> blob_ptr;
  CHECK(has_blob(blob_name)) << "Unknown blob name " << blob_name;
  blob_ptr = blobs_[blob_names_index_.find(blob_name)->second];
  return blob_ptr;
}

shared_ptr<NetParameter> ReadTextNetParameterFromFile(const string& file) {
  shared_ptr<NetParameter> np(new NetParameter);
  ReadNetParamsFromTextFileOrDie(file, np.get());
  return np;
}

shared_ptr<NetParameter> ReadTextNetParameterFromBuffer(const char* buffer, int buffer_len) {
  shared_ptr<NetParameter> np(new NetParameter);
  CHECK(google::protobuf::TextFormat::ParseFromString(std::string(buffer, buffer_len), np.get()))
    << "Parse Text NetParameter from Buffer failed";
  return np;
}

shared_ptr<NetParameter> ReadBinaryNetParameterFromFile(const string& file) {
  shared_ptr<NetParameter> np(new NetParameter);
  ReadNetParamsFromBinaryFileOrDie(file, np.get());
  return np;
}

shared_ptr<NetParameter> ReadBinaryNetParameterFromBuffer(const char* buffer, int buffer_len) {
  using google::protobuf::uint8;
  shared_ptr<NetParameter> np(new NetParameter);
  google::protobuf::io::CodedInputStream ci(reinterpret_cast<const uint8*>(buffer), buffer_len);
  CHECK(np->ParseFromCodedStream(&ci)) << "Parse Binary NetParameter from Buffer failed";
  return np;
}

}  // namespace caffe
