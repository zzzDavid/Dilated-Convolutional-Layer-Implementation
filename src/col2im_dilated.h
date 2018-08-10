#ifndef COL2IM_DILATED_H
#define COL2IM_DILATED_H

void col2im_dilated_cpu(float* data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, int dilate_rate, float* data_im);

#ifdef GPU
void col2im_dilated_gpu(float *data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, int dilate_rate, float *data_im);
#endif
#endif
