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
#include "priorbox_layer.h"
}

__global__ void add_variances_kernel(const int count, const float* input_data, float *output_data)
{
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += blockDim.x * gridDim.x) {

        for (int j = 0; j < 4; ++j) {
            output_data[4 * index + j] = input_data[j];
        }

    }
}

__global__ void add_priorbox_kernel(const int count, const int h, const int w, const float offset, const float step_w,
                                    const float step_h, const int min_size_size, const float * min_sizes, const int max_size_size,
                                    const float * max_sizes, const int ar_size, const float* aspect_ratios, const int size, const int img_height,
                                    const int img_width, const int flag, float *output_data)
{
	int s = 0;
	int r = 0;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += blockDim.x * gridDim.x) {

        float center_x = ((index % w) + offset) * step_w;
        float center_y = ((int)(index / w) + offset) * step_h;
        float box_width, box_height;
        for ( s = 0; s < min_size_size; ++s) {
            float min_size_ = min_sizes[s];
            // first prior: aspect_ratio = 1, size = min_size
            box_width = box_height = min_size_;
            output_data[index * size + 0] = (center_x - box_width / 2.f) / img_width;
            // ymin
            output_data[index * size + 1] = (center_y - box_height / 2.f) / img_height;
            // xmax
            output_data[index * size + 2] = (center_x + box_width / 2.f) / img_width;
            // ymax
            output_data[index * size + 3] = (center_y + box_height / 2.f) / img_height;

            if (flag) {
                //CHECK_EQ(min_sizes_.size(), max_sizes_.size())
                float max_size_ = max_sizes[s];
                // second prior: aspect_ratio = 1, size = sqrt(min_size * max_size);
                box_width = box_height = sqrtf(min_size_ * max_size_);
                // xmin
                output_data[index * size + 4] = (center_x - box_width / 2.) / img_width;
                // ymin
                output_data[index * size + 5] = (center_y - box_height / 2.) / img_height;
                // xmax
                output_data[index * size + 6] = (center_x + box_width / 2.) / img_width;
                // ymax
                output_data[index * size + 7] = (center_y + box_height / 2.) / img_height;
            }

            // rest of priors
            for (r = 0; r < ar_size - 1; ++r) {
                float ar = aspect_ratios[r + 1];
                /*
                if (fabs(ar - 1.) < 1e-6) {
                    continue;
                }
                 */
                box_width = min_size_ * sqrtf(ar);
                box_height = min_size_ / sqrtf(ar);
                // xmin
                output_data[index * size + 3 + flag * 4 + r * 4 + 1] = (center_x - box_width / 2.) / img_width;
                // ymin
                output_data[index * size + 3 + flag * 4 + r * 4 + 2] = (center_y - box_height / 2.) / img_height;
                // xmax
                output_data[index * size + 3 + flag * 4 + r * 4 + 3] = (center_x + box_width / 2.) / img_width;
                // ymax
                output_data[index * size + 3 + flag * 4 + r * 4 + 4] = (center_y + box_height / 2.) / img_height;
            }
        }

    }
}

void forward_priorbox_layer_gpu(priorbox_layer layer, network net)
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

    //int dim = layer.h * layer.w * layer.num_priors * 4;
    //float * top_data = calloc(2*layer.h*layer.w*layer.num_priors*4, sizeof(float));

    int h, w, s;
    int nthreads = layer.h * layer.w;
    int flag = 0;

    if (layer.max_size_size > 0 && layer.max_sizes != NULL)
        flag = 1;

    add_priorbox_kernel<<<(nthreads+BLOCK-1)/BLOCK, BLOCK>>>(
            nthreads, layer.h, layer.w, offset, step_w, step_h, layer.min_size_size, layer.min_sizes_gpu, layer.max_size_size,
                    layer.max_sizes_gpu, layer.ar_size, layer.aspect_ratios_gpu, layer.num_priors * 4, img_height, img_width, flag, layer.output_gpu);

    // set the variance.
    
    
    int start_index = layer.out_h * layer.out_w;
    float *v = layer.output_gpu + start_index;
    //layer.output_gpu += layer.out_h*layer.out_w;

    nthreads = layer.h * layer.w * layer.num_priors;

    add_variances_kernel<<<(nthreads+BLOCK-1)/BLOCK, BLOCK>>>(nthreads, layer.variances_gpu, v);

}
