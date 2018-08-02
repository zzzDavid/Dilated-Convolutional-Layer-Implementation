#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "im2col_dilated.h"
#include "cuda.h"
}

// src: https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu
// You may also want to read: https://github.com/BVLC/caffe/blob/master/LICENSE

__global__ void im2col_dilated_gpu_kernel(const int n, const float* im_gpu,
        const int height, const int width, const int ksize,
        const int pad,
        const int stride,
        const int height_col, const int width_col, int dilate_rate,
        float *col_gpu)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    for(; index < n; index += blockDim.x*gridDim.x){ 
        int w_out = index % width_col;     
        int h_index = index / width_col;   
        int h_out = h_index % height_col;
        int channel_in = h_index / height_col;
        int channel_out = channel_in * ksize * ksize;
        int h_in = h_out * stride - pad;   // height offset
        int w_in = w_out * stride - pad;   // width offset
        float* data_col_ptr = col_gpu;
        data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;//data_col_ptr += channel_out * width_col * height_col + h_out * width_col + w_out
        const float* data_im_ptr = im_gpu;
        data_im_ptr += (channel_in * height + h_in) * width + w_in;//data_im_ptr += channel_in * height * width + h_in * width + w_in
        for (int i = 1; i <= ksize; ++i) {
            for (int j = 1; j <= ksize; ++j) {
                int h = h_in + i;
                int w = w_in + j;

                *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ? data_im_ptr[i * width * dilate_rate -1 + j * dilate_rate -1] : 0;
                
                data_col_ptr += height_col * width_col; // 从这里看出这里是一列一列写，因此每次写一个window selection function的区域
            }
        }
    }
}

void im2col_dilated_gpu(float *im_cpu,
         int channels, int height, int width,
         int ksize, int stride, int pad, int dilate_rate, float *col_cpu){
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    
    int dilate_ksize = (dilate_rate - 1) * (ksize + 1) + ksize;
    int height_col = (height + 2 * pad - dilate_ksize) / stride + 1; // convolutional layer output height
    int width_col = (width + 2 * pad - dilate_ksize) / stride + 1;   // convolutional layer output width
    int num_kernels = channels * height_col * width_col;             // number of elements in each kernel

    
    // 在GPU分配内存
    float *im_gpu, *col_gpu;
    im_gpu = cuda_make_array(im_cpu, channels*height*width);
    col_gpu = cuda_make_array(col_cpu, num_kernels*height_col*width_col);
    

    im2col_dilated_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK,          //参数1：一个gird里有这么多block, 参数2：一个block里有这么多thread
       BLOCK>>>(num_kernels, im_gpu, height, width, ksize, pad,stride, height_col,width_col, dilate_rate, col_gpu);

    cudaMemcpy((void*)col_cpu, (void*)col_gpu, num_kernels*height_col*width_col*sizeof(float), cudaMemcpyDeviceToHost);
    // 释放内存
    cudaFree(im_gpu);
    cudaFree(col_gpu);
}

