//
//  permute_layer.c
//  darknet-xcode
//
//  Created by Tony on 2017/7/26.
//  Copyright © 2017年 tony. All rights reserved.
//

#include <stdio.h>
#include <stdbool.h>
#include "permute_layer.h"

void permute(const int count, float* input_data, const bool need_forward, const int *permute_orders,
             const int* old_steps, const int* new_steps, const int num_axes, float *output_data)
{
    int i,j;
    for (i=0; i<count; ++i) {
        int old_idx = 0;
        int idx = i;
        for (j=0; j<num_axes; ++j) {
            int order = permute_orders[j];
            old_idx += (idx / new_steps[j]) * old_steps[order];
            idx %= new_steps[j];
        }

        if (need_forward) {
            output_data[i] = input_data[old_idx];
        } else {
            input_data[old_idx] = output_data[i];
        }
    }

}

permute_layer make_permute_layer(int * permute_orders, int num_axes, bool need_permute, int h, int w, int c, int batch)
{
    permute_layer l = {0};
    l.type = PERMUTE;

    l.h = h;
    l.w = w;
    l.c = c;
    l.batch = batch;

    l.forward = forward_permute_layer;

    l.permute_orders = permute_orders;
    l.num_axes = num_axes;
    l.need_permute = need_permute;
    //l.out_c = l.c;
    l.outputs = h*w*c;

    l.output = calloc(l.batch*l.outputs,sizeof(float));

    int shapes[4] = {batch, c, h, w};

    l.batch = shapes[permute_orders[0]];
    l.out_c = shapes[permute_orders[1]];
    l.out_h = shapes[permute_orders[2]];
    l.out_w = shapes[permute_orders[3]];

#ifdef GPU
    l.forward_gpu = forward_permute_layer_gpu;

    if(gpu_index >= 0) {
        l.output_gpu = cuda_make_array(l.output, l.batch*l.outputs);
        int old_steps[4] = {l.c*l.h*l.w, l.h*l.w, l.w, 1};
        int new_steps[4] = {l.out_c*l.out_h*l.out_w, l.out_h*l.out_w, l.out_w, 1};
        l.permute_old_steps_gpu = cuda_make_int_array(old_steps, 4);
        l.permute_new_steps_gpu = cuda_make_int_array(new_steps, 4);
        l.permute_orders_gpu = cuda_make_int_array(l.permute_orders, 4);
    }
#endif

    fprintf(stderr, "Permute ");
    srand(0);

//    fprintf(stderr, "permute  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);
    
    return l;
}


void test_permute()
{
    
    int batch = 1;
    int c = 3;
    int h = 4;
    int w = 5;
    
    int out_c = h;
    int out_h = w;
    int out_w = c;
    
    int num_axis = 4;
    int permute_orders[4] = {0, 2, 3, 1};
    int outputs = c * h * w;
    
    const int top_count = batch * outputs;
    
    float *test_input = (float *)malloc(top_count * sizeof(float));
    float *test_output = (float *)malloc(top_count * sizeof(float));
    
    int i, j, k, m;
    printf("permute input:\n");
    for (i = 0; i < batch; ++i) {
        for (j = 0; j < c; ++j) {
            for (k = 0 ; k < h; ++k) {
                for (m = 0; m < w; m++) {
                    int index = ((i*c + j) * h + k) *w + m;
                    test_input[index] = i + j;
                    printf("%f ", test_input[index]);
                }
                printf("\n");
            }
            printf("\n\n");
        }
        printf("\n\n");
    }
    int old_steps[4] = {c*h*w, h*w, w, 1};
    
    int new_steps[4] = {out_c*out_h*out_w, out_h*out_w, out_w, 1};
    
    //for(i = 0; i < layer.num_axes; i++) {
    //    new_steps[i] = old_steps[layer.permute_orders[i]];
    //}
    const bool need_forward = true;
    permute(top_count, test_input, need_forward, permute_orders, old_steps,
            new_steps, 4, test_output);
    
    
    printf("permute output:\n");
    for (i = 0; i < batch; ++i) {
        for (j = 0; j < out_c; ++j) {
            for (k = 0 ; k < out_h; ++k) {
                for (m = 0; m < out_w; m++) {
                    int index = ((i*out_c + j) * out_h + k) *out_w + m;
                    printf("%f ", test_output[index]);
                }
                printf("\n");
            }
            printf("\n\n");
        }
        printf("\n\n");
    }

    
}

void forward_permute_layer(permute_layer layer, network net)
{
//    test_permute();
    if (layer.need_permute) {
        int i;
        const int top_count = layer.batch * layer.outputs;
        
        int old_steps[4] = {layer.c*layer.h*layer.w, layer.h*layer.w, layer.w, 1};
        
        int new_steps[4] = {layer.out_c*layer.out_h*layer.out_w, layer.out_h*layer.out_w, layer.out_w, 1};
        
        //for(i = 0; i < layer.num_axes; i++) {
        //    new_steps[i] = old_steps[layer.permute_orders[i]];
        //}
        const bool need_forward = true;
        permute(top_count, net.input, need_forward, layer.permute_orders, old_steps,
                new_steps, layer.num_axes, layer.output);
    } else {
        // If there is no need to permute, we share data to save memory.
        memcpy(layer.output, net.input, layer.outputs * layer.batch * sizeof(float));//   net.input;
    }
}

/*
void backward_permute_layer(const permute_layer layer) {
if (layer.need_permute) {
Dtype* top_diff = top[0]->mutable_cpu_diff();
Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
const int top_count = top[0]->count();
const int* permute_order = permute_order_.cpu_data();
const int* old_steps = old_steps_.cpu_data();
const int* new_steps = new_steps_.cpu_data();
bool forward = false;
Permute(top_count, bottom_diff, forward, permute_order, old_steps,
        new_steps, num_axes_, top_diff);
} else {
// If there is no need to permute, we share diff to save memory.
bottom[0]->ShareDiff(*top[0]);
}
}
*/
