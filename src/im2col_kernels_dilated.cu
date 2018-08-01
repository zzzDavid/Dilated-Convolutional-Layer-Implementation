#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "im2col_dilated.h"
#include "cuda.h"
}

// src: https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu
// You may also want to read: https://github.com/BVLC/caffe/blob/master/LICENSE

__global__ void im2col_dilated_gpu_kernel(const int n, const float* data_im,
        const int height, const int width, const int ksize,
        const int pad,
        const int stride,
        const int height_col, const int width_col, int dilate_rate,
        float *data_col)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;  // thread is 1-dimensional, block is 1-dimensional, Thread Index
                                                    // 计算出当前thread的index
    //printf("launch kernel success! I am kernel # %d. blockIndex = %d, threadIndex = %d\n", index, blockIdx.x, threadIdx.x);
    for(; index < n; index += blockDim.x*gridDim.x){ // blockDim.x = how many threads in each block; gridDim.x = how many blocks in each grid
                                                     // blockDim.x * gridDim.x = how many threads in each grid
                                                     // n = number of elements in each kernel
        //printf("n = %d, blockDim.x*gridDim.x = %d\n", n, blockDim.x*gridDim.x);
        int w_out = index % width_col;     // 
        int h_index = index / width_col;   // height index
        int h_out = h_index % height_col;  // 
        int channel_in = h_index / height_col; // channel in
        int channel_out = channel_in * ksize * ksize; // channel out
        int h_in = h_out * stride - pad;   // height offset
        int w_in = w_out * stride - pad;   // width offset
        //printf("w_out = %d, h_index = %d, hout = %d, channel_in = %d, channel_out  = %d, h_in = %d, w_in = %d\n", w_out, h_index, h_out, channel_in, channel_out, h_in, w_in);
        float* data_col_ptr = data_col;
        data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out; // 指向输出图像的指针
        const float* data_im_ptr = data_im;
        data_im_ptr += (channel_in * height + h_in) * width + w_in; //指向输入图片的指针
        printf("ksize = %d\n",ksize);
        for (int i = 1; i <= ksize; ++i) {
            for (int j = 1; j <= ksize; ++j) {
                int h = h_in + i;
                int w = w_in + j;
                printf("kernel index = %d, pixel selected: row = %d, column = %d, h = %d, w = %d\n",index, i, j, h, w);

                *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
                    data_im_ptr[i * width * dilate_rate -1 + j * dilate_rate -1] : 0;
                
                
                data_col_ptr += height_col * width_col; // 从这里看出这里是一列一列写，因此每次写一个window selection function的区域
            }
        }
    }
    //printf("kernel exited successfully!\n");
}

void im2col_dilated_gpu(float *im,
         int channels, int height, int width,
         int ksize, int stride, int pad, int dilate_rate, float *data_col){
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    printf("I'm in im2col_dilated_gpu!\n");
    int dilate_ksize = (dilate_rate - 1) * (ksize + 1) + ksize;
    int height_col = (height + 2 * pad - dilate_ksize) / stride + 1; // convolutional layer output height
    int width_col = (width + 2 * pad - dilate_ksize) / stride + 1;   // convolutional layer output width
    int num_kernels = channels * height_col * width_col;      // number of elements in each kernel
    printf("I'm going to launch kernel!\n");
    /*im2col_dilated_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK,          //参数1：一个gird里有这么多block, 参数2：一个block里有这么多thread
        BLOCK>>>(
                num_kernels, im, height, width, ksize, pad,
                stride, height_col,
                width_col, dilate_rate, data_col);*/
    // 我自己的试验：
    im2col_dilated_gpu_kernel<<<1,1>>>(num_kernels, im, height, width, ksize, pad, stride, height_col, width_col, dilate_rate, data_col);
    // 发射4个block，每个block里有4个kernel
}

