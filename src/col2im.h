#ifndef COL2IM_H
#define COL2IM_H

void col2im_dilated_cpu(float* data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_im, int dilate_rate);

#ifdef GPU
void col2im_dilated_gpu(float *data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, float *data_im, int dilate_rate);
#endif
#endif
