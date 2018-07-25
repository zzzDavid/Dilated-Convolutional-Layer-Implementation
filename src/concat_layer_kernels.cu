//
//  concat_layer.c
//  darknet-xcode
//
//  Created by Tony on 2017/7/26.
//  Copyright © 2017年 tony. All rights reserved.
//
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "im2col.h"
#include "cuda.h"
#include <stdio.h>
#include <stdbool.h>
#include "concat_layer.h"
}

__global__ void concat_gpu_kernel(const int count, const float* input_data, const bool kForward,
                                  const int num_concats, const int concat_size, const int top_concat_axis, const int bottom_concat_axis,
                                  const int offset_concat_axis, float *output_data)
{
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += blockDim.x * gridDim.x) {
        const int total_concat_size = concat_size * bottom_concat_axis;
        const int concat_num = index / total_concat_size;
        const int concat_index = index % total_concat_size;
        const int top_index = concat_index +
                              (concat_num * top_concat_axis + offset_concat_axis) * concat_size;
        if (kForward) {
            output_data[top_index] = input_data[index];
        } else {
            output_data[index] = input_data[top_index];
        }
    }
}

void forward_concat_layer_gpu(concat_layer layer, network net)
{
//    concat(net, layer.input_layers, layer.num_input_layers, layer.concat_axis, layer.output);
    int i;
    if (layer.num_input_layers == 1) return;
    int offset_concat_axis = 0;
    int bottom_concat_axis = 0;
    const bool kForward = true;
    for (i = 0; i < layer.num_input_layers; ++i) {
        int index = layer.input_layers[i];
        concat_layer l = net.layers[index];
        const float * bottom_data = l.output_gpu; //net.input;

        switch(layer.concat_axis) {
            case 0: bottom_concat_axis = l.batch; break;
            case 1: bottom_concat_axis = l.out_c; break;
            case 2: bottom_concat_axis = l.out_h; break;
            case 3: bottom_concat_axis = l.out_w; break;
            default: printf("Axis error in concat forward!\n");
        }
        const int bottom_concat_size = bottom_concat_axis * layer.concat_input_size;
        const int nthreads = bottom_concat_size * layer.num_concats;


        concat_gpu_kernel<<<(nthreads+BLOCK-1)/BLOCK, BLOCK>>>(
                nthreads, bottom_data, kForward, layer.num_concats, layer.concat_input_size,
                        layer.top_concat_axis, bottom_concat_axis, offset_concat_axis, layer.output_gpu);

        offset_concat_axis += bottom_concat_axis;
    }
}
