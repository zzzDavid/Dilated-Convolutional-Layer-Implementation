#include "softmax_layer.h"
#include "blas.h"
#include "cuda.h"
#include "gemm.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

softmax_layer make_softmax_layer(int *input_layer_indexes, int classes, int axis, int batch, int c, int h, int w, int inputs, int groups)
{
    assert(inputs%groups == 0);
    fprintf(stderr, "Softmax inputs = %d ",  inputs);
    softmax_layer l = {0};
    l.type = SOFTMAX;
    l.batch = batch;
    l.groups = groups;
    l.inputs = inputs;
    l.outputs = inputs;
    l.output = calloc(inputs*batch, sizeof(float));
    l.delta = calloc(inputs*batch, sizeof(float));

    l.input_layers = input_layer_indexes;
    l.axis = axis;
    l.classes = classes;
    l.h = classes;
    l.c = c*h*w/classes;
    l.w = 1;
    int shapes[4] = {l.batch, l.c, l.h, l.w};
    int i;
    int inner_num = 1;
    int outer_num = 1;
    for (i = 0; i < axis; i++) {
        outer_num *= shapes[i];
    }
    for (i = axis+1; i < 4; i++) {
        inner_num *= shapes[i];
    }
    l.out_c = outer_num;
    l.out_h = shapes[axis];
    l.out_w = inner_num;
    
    
    l.forward = forward_softmax_layer;
    l.backward = backward_softmax_layer;
    //printf("shapes = %d,%d,%d,%d; classes = %d; l.inner_num = %d; l.outer_num = %d; l.channels = %d; l.axis = %d;\n", l.batch, l.c, l.h, l.w, classes, l.inner_num, l.outer_num, l.channels, l.axis);
#ifdef GPU
    l.forward_gpu = forward_softmax_layer_gpu;
    l.backward_gpu = backward_softmax_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch);
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch);
#endif
    return l;
}

void forward_softmax_layer(softmax_layer l, network net)
{
//    int i, j, k;
//    float * scale_data = calloc(l.out_w, sizeof(float));
//    int dim = l.out_h * l.out_w;
//    
//    layer input_layer = net.layers[l.input_layers[0]];
//    float *input = input_layer.output;
//    memcpy(l.output, input, l.batch*l.outputs * sizeof(float));
//    float * sum_multiplier = calloc(l.out_h, sizeof(float));
//    for(i=0; i<l.out_h; i++) {
//        sum_multiplier[i] = 1;
//    }
//    // We need to subtract the max to avoid numerical issues, compute the exp,
//    // and then normalize.
////    printf("softmax output : \n");
//    int start_index = 0;
//    for (i = 0; i < l.out_c; ++i) {
//        // initialize scale_data to the first plane
//        memcpy(scale_data, input + i * dim, l.out_w * sizeof(float));
////        float * temp = input + i * dim;
//        start_index = i * dim;
//        for (j = 0; j < l.out_h; j++) {
//            for (k = 0; k < l.out_w; k++) {
//                scale_data[k] = (scale_data[k] > input[i * dim + j * l.out_w + k] ? scale_data[k] : input[i * dim + j * l.out_w + k]);
////                printf("scale_data[0] = %f; input = %f; j = %d; k = %d;\n", scale_data[0], input[i * dim + j * l.inner_num + k], j, k);
//            }
//        }
//        // subtraction
//        //axpy_cpu(l.channels*l.inner_num, -1., scale_data, 1, l.output, 1);
//        
//        gemm(0, 0, l.out_h, l.out_w, 1, -1., sum_multiplier, 1, scale_data, l.out_w, 1., l.output + start_index, l.out_w);////////
//        // exponentiation
//        for(j=0; j<dim; j++) {
//            //printf("exp l.output[%d] = %f\n", i*dim+j, exp(l.output[j + start_index]));
//            l.output[start_index + j] = exp(l.output[start_index + j]);
//        }
//        
//        gemm(1, 0, l.out_w, 1, l.out_h, 1., l.output + start_index, l.out_w, sum_multiplier, 1, 0., scale_data, 1);
//        // division
//        for (j = 0; j < l.out_h; j++) {
//            for(k = 0; k < l.out_w; k++) {
////                printf("scale_data[%d] = %f; m = %d;\n", n, scale_data[n], m);
//                l.output[start_index + j * l.out_w + k] /= scale_data[k];
////                if(l.output[start_index + j * l.out_w + k] > 0.25 && j != 0) printf("div l.output[%d] = %f\n", j, l.output[start_index + j * l.out_w + k]);
////                printf(" %f    %f \n", input[i * l.channels * l.inner_num + m * l.inner_num + n], l.output[n]);
//            }
//            
//            //caffe_div(inner_num_, top_data, scale_data, top_data);
////            l.output += l.inner_num;
//        }
//        
//        //printf("\n");
//    }

    
    if(l.softmax_tree){
        int i;
        int count = 0;
        for (i = 0; i < l.softmax_tree->groups; ++i) {
            int group_size = l.softmax_tree->group_size[i];
            softmax_cpu(net.input + count, group_size, l.batch, l.inputs, 1, 0, 1, l.temperature, l.output + count);
            count += group_size;
        }
    } else {
        layer input_layer = net.layers[l.input_layers[0]];
        float *input = input_layer.output;
        softmax_cpu(input, l.inputs/l.out_c, l.batch, l.inputs, l.out_c, l.inputs/l.out_c, 1, l.temperature, l.output);
    }
    
}

void backward_softmax_layer(const softmax_layer l, network net)
{
    axpy_cpu(l.inputs*l.batch, 1, l.delta, 1, net.delta, 1);
}

#ifdef GPU

void pull_softmax_layer_output(const softmax_layer layer)
{
    cuda_pull_array(layer.output_gpu, layer.output, layer.inputs*layer.batch);
}

void forward_softmax_layer_gpu(const softmax_layer l, network net)
{
    if(l.softmax_tree){
        int i;
        int count = 0;
        for (i = 0; i < l.softmax_tree->groups; ++i) {
            int group_size = l.softmax_tree->group_size[i];
            softmax_gpu(net.input_gpu + count, group_size, l.batch, l.inputs, 1, 0, 1, l.temperature, l.output_gpu + count);
            count += group_size;
        }
    } else {
        if(l.spatial){
            softmax_gpu(net.input_gpu, l.c, l.batch*l.c, l.inputs/l.c, l.w*l.h, 1, l.w*l.h, 1, l.output_gpu);
        }else{
			layer input_layer = net.layers[l.input_layers[0]];
			float *input_gpu = input_layer.output_gpu;
            softmax_gpu(input_gpu, l.inputs/l.out_c, l.batch, l.inputs, l.out_c, l.inputs/l.out_c, 1, l.temperature, l.output_gpu);
        }
    }
}

void backward_softmax_layer_gpu(const softmax_layer layer, network net)
{
    axpy_gpu(layer.batch*layer.inputs, 1, layer.delta_gpu, 1, net.delta_gpu, 1);
}

#endif
