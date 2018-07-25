
#ifndef __CUDA_IMAGE_H
#define __CUDA_IMAGE_H


#include "cudaUtility.h"

#include <stdint.h>


/**
 * Convert image data float RGB data to unsigned char data for cvMat
 * @ingroup util
 */
cudaError_t cudaImageRGBToMatData( float* srcDev, uchar3* destDev, size_t step, size_t width, size_t height);

cudaError_t cudaLetterboxImage(const float *im_data, const int w, const int h, float *output, const int net_w, const int net_h);

cudaError_t cudaCropImage(const float *im_data, const int w, const int h, float *output, int dx, int dy, int new_w, int new_h, const int channel);

//~ cudaError_t cudaImageResizeWidth(float* srcImage, float* dstImage, uint32_t srcw, uint32_t dstw, uint32_t srch, uint32_t srcc, float w_scale);

//~ cudaError_t cudaImageResizeHeight(float* srcImage, float* dstImage, uint32_t srch, uint32_t dsth, uint32_t srcw, uint32_t srcc, float h_scale); 

void resize_image_gpu(float* srcImage, float* dstImage, float *tmpImage, uint32_t srcw, uint32_t srch, uint32_t dstw, uint32_t dsth);

#endif
