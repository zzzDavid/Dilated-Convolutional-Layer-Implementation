//
//  flatten_layer.h
//  darknet-xcode
//
//  Created by Tony on 2017/7/26.
//  Copyright © 2017年 tony. All rights reserved.
//

#ifndef FLATTEN_LAYER_H
#define FLATTEN_LAYER_H

#include "cuda.h"
#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"

typedef layer flatten_layer;

/**
 * @brief Permute the input blob by changing the memory order of the data.
 *
 * TODO(weiliu89): thorough documentation for Forward, Backward, and proto params.
 */

// The main function which does the permute.

flatten_layer make_flatten_layer(int start_axis, int h, int w, int c, int batch);
void forward_flatten_layer(flatten_layer layer, network net);
//void backward_permute_layer(const permute_layer layer);

#ifdef GPU

void forward_flatten_layer_gpu(flatten_layer layer, network net);

#endif

#endif /* flatten_layer_h */
