#ifndef COMPILE_NET_UTIL_HPP_
#define COMPILE_NET_UTIL_HPP_
#include "../proto/caffe.pb.h"

namespace caffe {
/**
 *  @brief If CompileNet's compilation rule one does work, some scale layer's weights and bias blobs
 *  may be merged into batch norm layer. RecoverScaleFromBN will recover the merged scale layer's info.
 *  Currently, we only care about the weights and bias info.
 */
void RecoverScaleFromBN(const LayerParameter& bn_layer_param, LayerParameter& scale_layer_param, real_t default_scale_weights, real_t default_scale_bias);
/**
 *  @brief rename layer1's top to layer2's
 */
void MergeLayer(LayerParameter &layer1, const LayerParameter &layer2);

/**
 *  @brief After removing the batch norm and scale layer after a convolution layer, to make the inference
 *  result correct, we must adjust convolution layer's weights and bias blobs
 */

void AdjustConvLayer(LayerParameter &conv_layer,
                     const LayerParameter &batch_norm_layer,
                     const LayerParameter &scale_layer, bool is_net_init);

/**
 *  @brief The batch norm and scale layer may be merged due to compilation rule one's effect, RecoverBNScaleMergedNet
 *  is used to recover the scale layer
 */
void RecoverBNScaleMergedNet(NetParameter * net_param, NetParameter* recovered_net_param);

void RemoveBNScale(const NetParameter& param, NetParameter* param_compiled);
}
#endif
