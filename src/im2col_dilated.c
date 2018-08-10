#include "im2col.h"
#include <stdio.h>

float im2col_get_pixel(float *im, int height, int width, int channels, int row, int col, int channel, int pad);

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_dilated_cpu(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col, int dilate_rate) 
{
    printf("Entering im2col_dilated_cpu\n");
    int c,h,w;
    int dilate_ksize = (dilate_rate - 1) * (ksize + 1) + ksize;
    int height_col = (height + 2*pad - dilate_ksize) / stride + 1;
    int width_col = (width + 2*pad - dilate_ksize) / stride + 1;
    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 1; h <= height_col; ++h) {
            for (w = 1; w <= width_col; ++w) {
                int im_row = h_offset * dilate_rate + h * stride;
                int im_col = w_offset * dilate_rate + w * stride;
                int col_index = (c * height_col + h) * width_col + w - (width_col+1);
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }
}

