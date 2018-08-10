#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "col2im_dilated.h"
#include "cuda.h"
}

// src: https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu
// You may also want to read: https://github.com/BVLC/caffe/blob/master/LICENSE

__device__ float get_col_gpu_pixel(int row, int dilate_ksize, int ksize, int dilate_rate, int height_col, int width_col, int stride, int h_col, int w_col, const float* col_gpu)
{
    int width_kernel_dilated = row % dilate_ksize; // start from 1
    int height_kernel_dilated = (row / (dilate_ksize)) % dilate_ksize + 1; // start from 1
    if (width_kernel_dilated == 0){
    	width_kernel_dilated = dilate_ksize;
    	height_kernel_dilated--;
    }
    int channel_kernel_dilated = row / (dilate_ksize * dilate_ksize);  // start from 1
    int c = channel_kernel_dilated;
    int w = width_kernel_dilated / dilate_rate;
    int h = height_kernel_dilated / dilate_rate;

    int pixel_row = c * ksize * ksize + (h-1) * ksize + w - 1;
    int pixel_column = h_col * width_col + w_col;
    float pixel = col_gpu[pixel_row * width_col * height_col + pixel_column];
    return pixel;
}
__device__ bool isvalid(int dilate_ksize, int dilate_rate, int row)
{
    int width_kernel = row % dilate_ksize; // start from 1
    int height_kernel = (row / dilate_ksize)  % dilate_ksize + 1; // start from 1
    if (width_kernel == 0){
    	width_kernel = dilate_ksize;
    	height_kernel = height_kernel - 1;
    }
    if (width_kernel % dilate_rate==0 && height_kernel % dilate_rate==0) return 1;
    else return 0;
}

__global__ void col2im_dilated_gpu_kernel(const int n, const float* col_gpu,
        const int height, const int width, const int ksize,
        const int pad,
        const int stride,
        const int height_col, const int width_col, int dilate_rate,int channels,
        float *im_gpu) {
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    for(; index < n; index += blockDim.x*gridDim.x){
        float val = 0;
        int w = index % width + pad;
        int h = (index / width) % height + pad;
        int c = index / (width * height);
        // compute the start and end of the output
        int d_ksize = (dilate_rate - 1) * (ksize + 1) + ksize; // dilated kernel size
        int w_col_start = (w < d_ksize) ? 0 : (w - d_ksize) / stride + 1;
        int w_col_end = min(w / stride + 1, width_col);
        int h_col_start = (h < d_ksize) ? 0 : (h - d_ksize) / stride + 1;
        int h_col_end = min(h / stride + 1, height_col);
        // equivalent implementation
        int d_offset =
        (c * d_ksize * d_ksize + h * d_ksize + w) * height_col * width_col;
        int d_coeff_h_col = (1 - stride * d_ksize * height_col) * width_col;
        int d_coeff_w_col = (1 - stride * height_col * width_col);
        for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
            for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
                int row = (d_offset + h_col * d_coeff_h_col + w_col * d_coeff_w_col)/(height_col*width_col*channels);
                if(isvalid(d_ksize, dilate_rate, row)){
                    val += get_col_gpu_pixel(row, d_ksize, dilate_rate, ksize, height_col, width_col, stride, h_col, w_col, col_gpu);
                }else{
                    val += 0;
                }
            }
        }
        im_gpu[index-1] += val;
    }
}

void col2im_dilated_gpu(float *col_cpu,
        int channels, int height, int width,
        int ksize, int stride, int pad, int dilate_rate, float *im_cpu){
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.

    int dilate_ksize = (dilate_rate - 1) * (ksize + 1) + ksize;
    int height_col = (height + 2 * pad - dilate_ksize) / stride + 1; // convolutional layer output height
    int width_col = (width + 2 * pad - dilate_ksize) / stride + 1;   // convolutional layer output width
    int num_kernels = channels * height_col * width_col;             // number of elements in each kernel

    // allocate memory in GPU
    float *im_gpu, *col_gpu;
    cudaMalloc((void**)&im_gpu, channels*height*width*sizeof(float));
    cudaMalloc((void**)&col_gpu, channels*ksize*ksize*height_col*width_col*sizeof(float));

    cudaMemcpy(col_gpu, col_cpu, channels*ksize*ksize*height_col*width_col*sizeof(float), cudaMemcpyHostToDevice);

    col2im_dilated_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
        BLOCK>>>(
                channels*height*width, col_gpu, height, width, ksize, pad,
                stride, height_col,
                width_col, dilate_rate,channels, im_gpu);

     cudaMemcpy(im_cpu, im_gpu, channels*height*width*sizeof(float), cudaMemcpyDeviceToHost);
}

