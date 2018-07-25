//
//  priorbox_layer.c
//  darknet-xcode
//
//  Created by Tony on 2017/7/19.
//  Copyright © 2017年 tony. All rights reserved.
//

#include "priorbox_layer.h"

priorbox_layer make_priorbox_layer(int h, int w, int c, int batch, float *min_sizes, int min_size_size,float *max_sizes, int max_size_size, float *aspect_ratios, int ar_size, int num_priors, float *variances, float step, float offset)
{
    priorbox_layer l = {0};
    l.type = PRIORBOX;

    l.h = h;
    l.w = w;
    l.c = c;
    l.batch = 1;

    l.forward = forward_priorbox_layer;

    //l.min_sizes = min_sizes;
    l.min_size_size = min_size_size;
    //l.max_sizes = max_sizes;
    l.max_size_size = max_size_size;
    //l.aspect_ratios = aspect_ratios;
    l.ar_size = ar_size;
    l.num_priors = num_priors;
    //l.variances = variances;
    l.step = step;
    l.offset = offset;

    l.batch = 1;
    l.out_c = 2;
    l.out_h = h * w * num_priors * 4;
    l.out_w = 1;

    l.outputs = l.out_c*l.out_h*l.out_w;
    l.output = calloc(l.batch * l.outputs, sizeof(float));
    int i;
    if(l.min_size_size > 0) 
    {
		l.min_sizes = calloc(l.min_size_size, sizeof(float));
		for(i = 0; i < l.min_size_size; ++i) l.min_sizes[i] = min_sizes[i];
	}
    if(l.max_size_size > 0) 
    {
		l.max_sizes = calloc(l.max_size_size, sizeof(float));
		for(i = 0; i < l.max_size_size; ++i) l.max_sizes[i] = max_sizes[i];
	}
    if(l.ar_size > 0) 
    {
		l.aspect_ratios = calloc(l.ar_size, sizeof(float));
		for(i = 0; i < l.ar_size; ++i) l.aspect_ratios[i] = aspect_ratios[i];
	}
    l.variances = calloc(4, sizeof(float));   
    for(i = 0; i < 4; ++i) l.variances[i] = variances[i];

#ifdef GPU
    l.forward_gpu = forward_priorbox_layer_gpu;

    if(gpu_index >= 0) {
        l.output_gpu = cuda_make_array(l.output, l.outputs * l.batch);
        if(l.min_size_size > 0) l.min_sizes_gpu = cuda_make_array(l.min_sizes, l.min_size_size);
        if(l.max_size_size > 0) l.max_sizes_gpu = cuda_make_array(l.max_sizes, l.max_size_size);
        if(l.ar_size > 0) l.aspect_ratios_gpu = cuda_make_array(l.aspect_ratios, l.ar_size);
        l.variances_gpu = cuda_make_array(l.variances, 4);
    }

#endif

    fprintf(stderr, "Priorbox ");
    srand(0);

    return l;
}

void forward_priorbox_layer(priorbox_layer layer, network net)
{
    float img_width = (float)net.w;
    float img_height = (float)net.h;
    float step_w;
    float step_h;

    if (layer.step == 0) {
        step_w = (float)(img_width) / layer.w;
        step_h = (float)(img_height) / layer.h;
    } else {
        step_w = layer.step;
        step_h = layer.step;
    }
    float offset = layer.offset;
    int idx = 0;

    //int dim = layer.h * layer.w * layer.num_priors * 4;
    //float * top_data = calloc(2*layer.h*layer.w*layer.num_priors*4, sizeof(float));

    int i, j, h, w, s, r;
    for ( h = 0; h < layer.h; ++h) {
        for ( w = 0; w < layer.w; ++w) {
            float center_x = (w + offset) * step_w;
            float center_y = (h + offset) * step_h;
            float box_width, box_height;
            for ( s = 0; s < layer.min_size_size; ++s) {
                int min_size_ = layer.min_sizes[s];
                // first prior: aspect_ratio = 1, size = min_size
                box_width = box_height = min_size_;
                layer.output[idx++] = (center_x - box_width / 2.f) / img_width;
                // ymin
                layer.output[idx++] = (center_y - box_height / 2.f) / img_height;
                // xmax
                layer.output[idx++] = (center_x + box_width / 2.f) / img_width;
                // ymax
                layer.output[idx++] = (center_y + box_height / 2.f) / img_height;
                if (layer.max_size_size > 0 && layer.max_sizes != NULL) {
                    //CHECK_EQ(min_sizes_.size(), max_sizes_.size())
                    int max_size_ = layer.max_sizes[s];
                    // second prior: aspect_ratio = 1, size = sqrt(min_size * max_size);
                    box_width = box_height = sqrtf(min_size_ * max_size_);
                    // xmin
                    layer.output[idx++] = (center_x - box_width / 2.) / img_width;
                    // ymin
                    layer.output[idx++] = (center_y - box_height / 2.) / img_height;
                    // xmax
                    layer.output[idx++] = (center_x + box_width / 2.) / img_width;
                    // ymax
                    layer.output[idx++] = (center_y + box_height / 2.) / img_height;
                }

                // rest of priors
                for (r = 0; r < layer.ar_size; ++r) {
                    float ar = layer.aspect_ratios[r];
                    if (fabs(ar - 1.) < 1e-6) {
                        continue;
                    }
                    box_width = min_size_ * sqrtf(ar);
                    box_height = min_size_ / sqrtf(ar);
                    // xmin
                    layer.output[idx++] = (center_x - box_width / 2.) / img_width;
                    // ymin
                    layer.output[idx++] = (center_y - box_height / 2.) / img_height;
                    // xmax
                    layer.output[idx++] = (center_x + box_width / 2.) / img_width;
                    // ymax
                    layer.output[idx++] = (center_y + box_height / 2.) / img_height;
                }
            }
        }
    }

    // set the variance.
//    layer.output += layer.out_h*layer.out_w;

    int start_index = layer.out_h*layer.out_w;
    int count = 0;
    for (h = 0; h < layer.h; ++h) {
        for (w = 0; w < layer.w; ++w) {
            for (i = 0; i < layer.num_priors; ++i) {
                for (j = 0; j < 4; ++j) {
                    layer.output[count + start_index] = layer.variances[j];
                    ++count;
                }
            }
        }
    }
    
}
