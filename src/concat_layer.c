#include "concat_layer.h"
#include <assert.h>

concat_layer make_concat_layer(int axis, int num, int batch, int * layers, int * sizes, network net)
{
    int i;
    concat_layer l = {0};

    l.type = CONCAT;

    layer first = net.layers[layers[0]];
    l.w = l.out_w = first.out_w;
    l.h = l.out_h = first.out_h;
    l.c = l.out_c = first.out_c;

    int shapes[4] = {first.batch, first.out_c, first.out_h, first.out_w};
    /*
    for(i = 1; i < num; ++i){
        int index = layers[i];
        layer next = net.layers[index];
        if(next.out_w == first.out_w && next.out_h == first.out_h){            /////////////////////???????????
            l.out_c += next.out_c;
        }else{
            l.out_h = l.out_w = l.out_c = 0;
        }
    }
     */

    l.forward = forward_concat_layer;
    l.concat_axis = axis;

    l.batch = batch;
    l.num_concats = 1;
    for (i = 0; i < axis; ++i) {
        l.num_concats *= shapes[i];
    }
    l.input_layers = layers;
    l.input_sizes = sizes;
    l.num_input_layers = num;
    l.concat_input_size = 1;
    for (i = axis+1; i < 4; ++i) {
        l.concat_input_size *= shapes[i];
    }

    fprintf(stderr, "Concat ");

    int outputs = 0;
    for(i = 0; i < num; ++i){
        fprintf(stderr," %d", layers[i]);
        outputs += sizes[i];
    }
    l.outputs = outputs;
    l.inputs = outputs;
    l.top_concat_axis = 0;

    if(l.num_input_layers == 1) l.output = first.output;
    else {
        l.output = calloc(outputs*batch, sizeof(float));
    }

    switch (axis) {
        case 0: {
            for (i = 0; i < num; i++) l.top_concat_axis += net.layers[layers[i]].batch;
            l.batch  = l.top_concat_axis;
            break;
        }
        case 1: {
            for (i = 0; i < num; i++) l.top_concat_axis += net.layers[layers[i]].out_c;
            l.out_c  = l.top_concat_axis;
            break;
        }
        case 2: {
            for (i = 0; i < num; i++) l.top_concat_axis += net.layers[layers[i]].out_h;
            l.out_h  = l.top_concat_axis;
            break;
        }
        case 3: {
            for (i = 0; i < num; i++) l.top_concat_axis += net.layers[layers[i]].out_w;
            l.out_w  = l.top_concat_axis;
            break;
        }
        default: printf("Axis error in concat layer!\n");
    }

#ifdef GPU
    l.forward_gpu = forward_concat_layer_gpu;

    if(gpu_index >= 0) {
        if(l.num_input_layers == 1) l.output_gpu = first.output_gpu;
        else {
            l.output_gpu = cuda_make_array(l.output, l.outputs * l.batch);
        }
    }
    
#endif


    srand(0);

    return l;
}


void concat(network net, int *input_layers_index, int num, int axis, float *output)
{
    assert(axis < 4 && axis >= 0);
    
    layer first =  net.layers[input_layers_index[0]];
    
    int i, j;
    
    int n = first.batch;
    int c = first.out_c;
    int h = first.out_h;
    int w = first.out_w;
    
    int inputs = n * c * h * w;
    int axises[4] = {n, c, h, w};
    
    int num_concats = 1;
    for (i = 0; i < axis; ++i) {
        num_concats *= axises[i];
    }
    int offset = inputs / num_concats;

    int output_step = num * offset;// assume each layer has the same offset
    for (i = 0; i < num; ++i) {
        int index = input_layers_index[i];
        layer l = net.layers[index];
        float *input = l.output;
        for (j = 0; j < num_concats; ++j) {
            memcpy(output + output_step * j + i * offset, input + j * offset, offset * sizeof(float));
        }
    }
}



void forward_concat_layer(concat_layer layer, network net)
{
//    concat(net, layer.input_layers, layer.num_input_layers, layer.concat_axis, layer.output);
    int i;
    if (layer.num_input_layers == 1) return;
    int offset_concat_axis = 0;
    int bottom_concat_axis = 0;
    for (i = 0; i < layer.num_input_layers; ++i) {
        int index = layer.input_layers[i];
        concat_layer l = net.layers[index];
        const float * bottom_data = l.output; //net.input;

        switch(layer.concat_axis) {
            case 0: bottom_concat_axis = l.batch; break;
            case 1: bottom_concat_axis = l.out_c; break;
            case 2: bottom_concat_axis = l.out_h; break;
            case 3: bottom_concat_axis = l.out_w; break;
            default: printf("Axis error in concat forward!\n");
        }

        int n;
        for (n = 0; n < layer.num_concats; ++n) {
            memcpy(layer.output + (n * layer.top_concat_axis + offset_concat_axis) * layer.concat_input_size,
                   bottom_data + n * bottom_concat_axis * layer.concat_input_size,
                   bottom_concat_axis * layer.concat_input_size * sizeof(float));
        }
        offset_concat_axis += bottom_concat_axis;
    }
}


/*
#ifdef GPU

void forward_concat_layer_gpu(concat_layer layer, network net)
{
    //    concat(net, layer.input_layers, layer.num_input_layers, layer.concat_axis, layer.output);
    int i;
    if (layer.num_input_layers == 1) return;
    int offset_concat_axis = 0;
    int bottom_concat_axis = 0;
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
        
        int n;
        for (n = 0; n < layer.num_concats; ++n) {
            int len = bottom_concat_axis * layer.concat_input_size;
            int input_offset = n * bottom_concat_axis * layer.concat_input_size;
            int output_offset = (n * layer.top_concat_axis + offset_concat_axis) * layer.concat_input_size;
            copy_gpu(len, bottom_data + input_offset, layer.output_gpu + output_offset, 1);
            
        }
        offset_concat_axis += bottom_concat_axis;
    }
}


#endif
*/
