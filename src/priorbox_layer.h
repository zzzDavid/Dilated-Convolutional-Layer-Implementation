//
//  priorbox_layer.h
//  darknet-xcode
//
//  Created by Tony on 2017/7/19.
//  Copyright © 2017年 tony. All rights reserved.
//

#ifndef PRIORBOX_LAYER_H
#define PRIORBOX_LAYER_H

#include "cuda.h"
#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"

typedef layer priorbox_layer;

/**
 * @brief Generate the prior boxes of designated sizes and aspect ratios across
 *        all dimensions @f$ (H \times W) @f$.
 *
 * Intended for use with MultiBox detection method to generate prior (template).
 *
 * NOTE: does not implement Backwards operation.
 */
priorbox_layer make_priorbox_layer(int h, int w, int c, int batch, float *min_sizes, int min_size_size,float *max_sizes, int max_size_size, float *aspect_ratios, int ar_size, int num_priors, float *variance, float step, float offset);
//void resize_convolutional_layer(convolutional_layer *layer, int w, int h);
void forward_priorbox_layer(priorbox_layer layer, network net);

#ifdef GPU

void forward_priorbox_layer_gpu(priorbox_layer layer, network net);

#endif

#endif /* priorbox_layer_h */
