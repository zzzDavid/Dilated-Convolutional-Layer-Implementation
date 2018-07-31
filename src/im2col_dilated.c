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
    //height_col = height_col / dilate_rate;
    //width_col = width_col / dilate_rate;
    //printf("height_col = width_col = %d\n", height_col);
    //printf("ksize = %d\n",ksize);

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
                /*printf("col_index = %d, data = %f\n",col_index, data_col[col_index]);
                printf("im_row = %d\t", im_row);
                printf("im_col = %d\t", im_col);
                printf("im_channel = %d\t", c_im);
                printf("h_offset = %d\t",h_offset);
                printf("w_offset = %d\t", w_offset);
                printf("c = %d\n",c);*/
            }
        }
    }
    
    /*
    printf("im2col output:\n");
    for (int i = 0; i < height_col*width_col*channels_col; i++)
    {
        if ((i+1) % 36 == 0)
        {
            printf("%d ", (int)data_col[i]);
            printf("\n");
        }else{
            printf("%d ", (int)data_col[i]);
        }
        //printf("i = %d\t", i);
    }*/
}

