#ifndef _CAFFE_UTIL_INSERT_SPLITS_HPP_
#define _CAFFE_UTIL_INSERT_SPLITS_HPP_

#include <string>
#include "../proto/caffe.pb.h"

namespace caffe {

// Copy NetParameters with SplitLayers added to replace any shared bottom
// blobs with unique bottom blobs provided by the SplitLayer.
void InsertSplits(const NetParameter& param, NetParameter* param_split);

void ConfigureSplitLayer(const std::string& layer_name, const std::string& blob_name,
    const int blob_idx, const int split_count, const float loss_weight,
    LayerParameter* split_layer_param);

std::string SplitLayerName(const std::string& layer_name, const std::string& blob_name,
    const int blob_idx);

std::string SplitBlobName(const std::string& layer_name, const std::string& blob_name,
    const int blob_idx, const int split_idx);

}  // namespace caffe

#endif  // CAFFE_UTIL_INSERT_SPLITS_HPP_
