#include <algorithm>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <caffe/logging.hpp>

#include "insert_splits.hpp"

namespace caffe {

void InsertSplits(const NetParameter& param, NetParameter* param_split) {
  // Initialize by copying from the input NetParameter.
  param_split->CopyFrom(param);
  param_split->clear_layer();
  std::map<std::string, std::pair<int, int> > blob_name_to_last_top_idx;
  std::map<std::pair<int, int>, std::pair<int, int> > bottom_idx_to_source_top_idx;
  std::map<std::pair<int, int>, int> top_idx_to_bottom_count;
  std::map<std::pair<int, int>, float> top_idx_to_loss_weight;
  std::map<std::pair<int, int>, int> top_idx_to_bottom_split_idx;
  std::map<int, std::string> layer_idx_to_layer_name;
  for (int i = 0; i < param.layer_size(); ++i) {
    const LayerParameter& layer_param = param.layer(i);
    layer_idx_to_layer_name[i] = layer_param.name();
    for (int j = 0; j < layer_param.bottom_size(); ++j) {
      const std::string& blob_name = layer_param.bottom(j);
      if (blob_name_to_last_top_idx.find(blob_name) ==
          blob_name_to_last_top_idx.end()) {
        LOG(FATAL) << "Unknown bottom blob '" << blob_name << "' (layer '"
                   << layer_param.name() << "', bottom index " << j << ")";
      }
      const std::pair<int, int>& bottom_idx = std::make_pair(i, j);
      const std::pair<int, int>& top_idx = blob_name_to_last_top_idx[blob_name];
      bottom_idx_to_source_top_idx[bottom_idx] = top_idx;
      ++top_idx_to_bottom_count[top_idx];
    }
    for (int j = 0; j < layer_param.top_size(); ++j) {
      const std::string& blob_name = layer_param.top(j);
      blob_name_to_last_top_idx[blob_name] = std::make_pair(i, j);
    }
    // A use of a top blob as a loss should be handled similarly to the use of
    // a top blob as a bottom blob to another layer.
    const int last_loss =
        std::min(layer_param.loss_weight_size(), layer_param.top_size());
    for (int j = 0; j < last_loss; ++j) {
      const std::string& blob_name = layer_param.top(j);
      const std::pair<int, int>& top_idx = blob_name_to_last_top_idx[blob_name];
      top_idx_to_loss_weight[top_idx] = layer_param.loss_weight(j);
      if (top_idx_to_loss_weight[top_idx]) {
        ++top_idx_to_bottom_count[top_idx];
      }
    }
  }
  for (int i = 0; i < param.layer_size(); ++i) {
    LayerParameter* layer_param = param_split->add_layer();
    layer_param->CopyFrom(param.layer(i));
    // Replace any shared bottom blobs with split layer outputs.
    for (int j = 0; j < layer_param->bottom_size(); ++j) {
      const std::pair<int, int>& top_idx =
          bottom_idx_to_source_top_idx[std::make_pair(i, j)];
      const int split_count = top_idx_to_bottom_count[top_idx];
      if (split_count > 1) {
        const std::string& layer_name = layer_idx_to_layer_name[top_idx.first];
        const std::string& blob_name = layer_param->bottom(j);
        layer_param->set_bottom(j, SplitBlobName(layer_name,
            blob_name, top_idx.second, top_idx_to_bottom_split_idx[top_idx]++));
      }
    }
    // Create split layer for any top blobs used by other layer as bottom
    // blobs more than once.
    for (int j = 0; j < layer_param->top_size(); ++j) {
      const std::pair<int, int>& top_idx = std::make_pair(i, j);
      const int split_count = top_idx_to_bottom_count[top_idx];
      if (split_count > 1) {
        const std::string& layer_name = layer_idx_to_layer_name[i];
        const std::string& blob_name = layer_param->top(j);
        LayerParameter* split_layer_param = param_split->add_layer();
        const float loss_weight = top_idx_to_loss_weight[top_idx];
        ConfigureSplitLayer(layer_name, blob_name, j, split_count,
            loss_weight, split_layer_param);
        if (loss_weight) {
          layer_param->clear_loss_weight();
          top_idx_to_bottom_split_idx[top_idx]++;
        }
      }
    }
  }
}

void ConfigureSplitLayer(const std::string& layer_name, const std::string& blob_name,
    const int blob_idx, const int split_count, const float loss_weight,
    LayerParameter* split_layer_param) {
  split_layer_param->Clear();
  split_layer_param->add_bottom(blob_name);
  split_layer_param->set_name(SplitLayerName(layer_name, blob_name, blob_idx));
  split_layer_param->set_type("Split");
  for (int k = 0; k < split_count; ++k) {
    split_layer_param->add_top(
        SplitBlobName(layer_name, blob_name, blob_idx, k));
    if (loss_weight) {
      if (k == 0) {
        split_layer_param->add_loss_weight(loss_weight);
      } else {
        split_layer_param->add_loss_weight(0);
      }
    }
  }
}

std::string SplitLayerName(const std::string& layer_name, const std::string& blob_name,
    const int blob_idx) {
  std::ostringstream split_layer_name;
  split_layer_name << blob_name << "_" << layer_name << "_" << blob_idx
      << "_split";
  return split_layer_name.str();
}

std::string SplitBlobName(const std::string& layer_name, const std::string& blob_name,
    const int blob_idx, const int split_idx) {
  std::ostringstream split_blob_name;
  split_blob_name << blob_name << "_" << layer_name << "_" << blob_idx
      << "_split_" << split_idx;
  return split_blob_name.str();
}

}  // namespace caffe
