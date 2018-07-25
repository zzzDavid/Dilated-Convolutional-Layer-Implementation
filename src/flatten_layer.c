//
//  flatten_layer.c
//  darknet-xcode
//
//  Created by Tony on 2017/7/26.
//  Copyright © 2017年 tony. All rights reserved.
//

#include "flatten_layer.h"

flatten_layer make_flatten_layer(int start_axis, int h, int w, int c, int batch)
{
    flatten_layer l = {0};
    l.type = FLATTEN;

    l.h = h;
    l.w = w;
    l.c = c;
    l.batch = batch;

    l.forward = forward_flatten_layer;

    l.start_axis = start_axis;
    l.output = calloc(l.h*l.w*l.c*l.batch,sizeof(float));
    //int shapes[4] = {batch, c, h, w};
    l.out_c = c*h*w;
    l.out_h = 1;
    l.out_w = 1;
    l.outputs = h*w*c;

#ifdef GPU
    l.forward_gpu = forward_flatten_layer_gpu;

    if(gpu_index >= 0) {
        l.output_gpu = cuda_make_array(l.output, l.outputs * l.batch);
    }

#endif

    fprintf(stderr, "Flatten ");
    srand(0);

    return l;
}

void forward_flatten_layer(flatten_layer layer, network net)
{
//    layer.output = net.input;
    memcpy(layer.output, net.input, layer.outputs * layer.batch * sizeof(float));// net.input;
    return;
}

#ifdef GPU

void forward_flatten_layer_gpu(flatten_layer layer, network net)
{
    copy_gpu(layer.outputs * layer.batch, net.input_gpu, 1, layer.output_gpu, 1);// net.input;
    return;
}

#endif
