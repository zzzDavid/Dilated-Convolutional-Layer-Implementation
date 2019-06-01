#include "im2col.h"
#include <stdio.h>

float im2col_get_pixel(float *im, int height, int width, int channels, int row, int col, int channel, int pad);

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_dilated_cpu(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col, int dilate_rate) 
{
    //printf("Entering im2col_dilated_cpu\n");
    int c,h,w;
    int dilate_ksize = (dilate_rate - 1) * (ksize + 1) + ksize;
    int height_col = (height + 2*pad - dilate_ksize) / stride + 1;
    int width_col = (width + 2*pad - dilate_ksize) / stride + 1;
    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize + 1;
        int h_offset = (c / ksize) % ksize + 1;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset * dilate_rate + h * stride - 1;
                int im_col = w_offset * dilate_rate + w * stride - 1;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
		//printf("im_row = %d, im_col = %d, pixel = %f\n", im_row, im_col, data_col[col_index]);

            }
        }
    }
}

