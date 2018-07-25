//
//  permute_layer.c
//  darknet-xcode
//
//  Created by Tony on 2017/7/26.
//  Copyright © 2017年 tony. All rights reserved.
//

#ifndef PERMUTE_LAYER_H
#define PERMUTE_LAYER_H

#include "cuda.h"
#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"

typedef layer permute_layer;

/**
 * @brief Permute the input blob by changing the memory order of the data.
 *
 * TODO(weiliu89): thorough documentation for Forward, Backward, and proto params.
 */

// The main function which does the permute.
void permute(const int count, float * input, const bool forward,
                 const int* permute_orders, const int* old_steps, const int* new_steps,
                 const int num_axes, float * output);

permute_layer make_permute_layer(int * permute_orders, int num_axes, bool need_permute, int h, int w, int c, int batch);
void forward_permute_layer(permute_layer layer, network net);
//void backward_permute_layer(const permute_layer layer);

#ifdef GPU

void forward_permute_layer_gpu(permute_layer layer, network net);

#endif

#endif  // PERMUTE_LAYER_H

