//
//  permute_layer.c
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
#include "permute_layer.h"
}

__global__ void permute_gpu_kernel(const int count, float* input_data, const bool need_forward, const int *permute_orders, const int* old_steps, const int* new_steps, const int num_axes, float *output_data)
{
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += blockDim.x * gridDim.x) {
        int temp_idx = index;
        int old_idx = 0;
        for (int i=0; i < num_axes; ++i)
        {
            int order = permute_orders[i];
            old_idx += (temp_idx / new_steps[i]) * old_steps[order];
            temp_idx %= new_steps[i];
        }

        if (need_forward) {
            output_data[index] = input_data[old_idx];
        } else {
            input_data[old_idx] = output_data[index];
        }
    }
}


void forward_permute_layer_gpu(permute_layer layer, network net)
{
    if (layer.need_permute) {
        //int i;
        const int top_count = layer.batch * layer.outputs;
        
        //int old_steps[4] = {layer.c*layer.h*layer.w, layer.h*layer.w, layer.w, 1};
        
        //int new_steps[4] = {layer.out_c*layer.out_h*layer.out_w, layer.out_h*layer.out_w, layer.out_w, 1};
        
        //for(i = 0; i < layer.num_axes; i++) {
        //    new_steps[i] = old_steps[layer.permute_orders[i]];
        //}
        const int *old_steps_gpu = layer.permute_old_steps_gpu;//cuda_make_int_array(old_steps, 4);
        const int *new_steps_gpu = layer.permute_new_steps_gpu;//cuda_make_int_array(new_steps, 4);
        const int *permute_orders_gpu = layer.permute_orders_gpu;//cuda_make_int_array(layer.permute_orders, 4);
        
        const bool need_forward = true;
        permute_gpu_kernel<<<(top_count+BLOCK-1)/BLOCK, BLOCK>>>(
                top_count, net.input_gpu, need_forward, permute_orders_gpu, old_steps_gpu, new_steps_gpu, layer.num_axes, layer.output_gpu);
    } else {
        // If there is no need to permute, we share data to save memory.
        layer.output_gpu = net.input_gpu;//   net.input;
    }
}
