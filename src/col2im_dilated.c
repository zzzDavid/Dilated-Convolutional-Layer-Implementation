#include <stdio.h>
#include <math.h>
#include "col2im.h"
#include "col2im_dilated.h"
void col2im_add_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad, float val);


void col2im_dilated_cpu(float* data_col,
         int channels,  int height,  int width,
         int ksize,  int stride, int pad, int dilate_rate, float* data_im)
{
    int c,h,w;
    int dilate_ksize = (dilate_rate - 1) * (ksize + 1) + ksize;
    int height_col = (height + 2*pad - dilate_ksize) / stride + 1;
    int width_col = (width + 2*pad - dilate_ksize) / stride + 1;


    int channels_col = channels * ksize * ksize;
    for (c = 1; c <= channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize+1;
        if (w_offset == 0){
        	w_offset = ksize;
        	h_offset--;
        }
        if (h_offset == 0) h_offset = ksize;
        int c_im = (c-1) / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset * dilate_rate + h * stride;
                int im_col = w_offset * dilate_rate + w * stride;
                int col_index = ((c-1) * height_col + h) * width_col + w;
                double val = data_col[col_index];
                //printf("im_row = %d, im_col = %d, val = %d\t location in window:(%d, %d)\n",im_row, im_col, (int)val, h_offset, w_offset);
                col2im_add_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad, val);
            }
        }
    }
}

