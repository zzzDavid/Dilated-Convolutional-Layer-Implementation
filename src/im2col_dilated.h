#ifndef IM2COL_DILATED_H
#define IM2COL_DILATED_H

void im2col_dilated_cpu(float* data_im,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_col, int dilate_rate);

#ifdef GPU

void im2col_dilated_gpu(float *im,
         int channels, int height, int width,
         int ksize, int stride, int pad, int dilate_rate, float *data_col);

#endif
#endif
