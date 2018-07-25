#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "darknet.h"
#include "cuda.h"
}

inline __device__ __host__ int iDivUp( int a, int b )  		{ return (a % b != 0) ? (a / b + 1) : (a / b); }

__global__ void letterbox_image_kernel( const float *input,
                                  float *output,
                                  const int width_in,
                                  const int height_in,
                                  const int width_resized,
                                  const int height_resized,                                  
                                  const int width_out,
                                  const int height_out)
{
    const float xScale = (float)(width_in * 1.f / width_resized);
    const float yScale = (float)(height_in * 1.f/height_resized);
    const int xIndex = blockIdx.x*blockDim.x+threadIdx.x;
    const int yIndex = blockIdx.y*blockDim.y+threadIdx.y;
    
    if(xIndex >= width_out || yIndex >= height_out)
        return;
    
    const int tid = yIndex * width_out + xIndex;
    const int offset_out = width_out * height_out;
    const int offset_resized = width_resized * height_resized;
    const int offset_in = width_in * height_in;
    
    
    
    //~ float3 A1 = input[intY * width_in + intX];
    //~ float3 A2 = input[intY * width_in + intX+1];
    //~ float3 A3 = input[(intY+1) * width_in + intX];
    //~ float3 A4 = input[(intY+1) * width_in + intX+1];
    const int dx = (width_out - width_resized)/2;
    const int dy = (height_out - height_resized)/2;
    
    if ((xIndex < dx || xIndex > width_resized + dx) ||(yIndex < dy || yIndex > height_resized + dy))
    {
	    output[tid] = 0.5f;
	    output[tid + offset_out] = 0.5f;
	    output[tid + offset_out * 2] = 0.5f;
    } else {
	    int x = xIndex - dx;
	    int y = yIndex - dy;
	    const float inXindex = (float)(x * 1.f * xScale) + 0.5f;
	    const float inYindex = (float)(y * 1.f * yScale) + 0.5f;
    
    
	    const int intX = (int)(inXindex + 0.5f);
	    const int intY = (int)(inYindex + 0.5f);
    
	    const float a = inXindex - intX + 0.5f;
	    const float b = inYindex - intY + 0.5f;
    
	    const int p1 = intY * width_in + intX;
	    const int p2 = intY * width_in + intX + 1;
	    const int p3 = (intY+1) * width_in + intX;
	    const int p4 = (intY+1) * width_in + intX + 1;
		
            output[tid] = (1-a) * (1-b) * input[p1] + a * (1-b) * input[p2] + (1-a)*b*input[p3] + a * b * input[p4];
	    output[tid + offset_out] = (1-a) * (1-b) * input[p1+offset_in] + a * (1-b) * input[p2 + offset_in] + (1-a)*b*input[p3 + offset_in] + a * b * input[p4 + offset_in];
	    output[tid + offset_out*2] = (1-a) * (1-b) * input[p1+offset_in*2] + a * (1-b) * input[p2 + offset_in * 2] + (1-a)*b*input[p3 + offset_in * 2] + a * b * input[p4 + offset_in * 2];
//                if(x < 10 && y < 10)
//	              printf("\n offset:(%d %d) out_index: (%d %d) in_index: (%d %d) scale:(%f %f) \n out(rgb): %f %f %f\n", dx, dy, xIndex, yIndex, intX, intY, xScale, yScale, p1, p2, p3, p4, output[tid],output[tid+offset_out], output[tid+offset_out*2]);
    }
}


void letterbox_image_into_gpu(const float *im_data, const int w, const int h, float *im_boxed, const int box_w, const int box_h)
{
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int resized_w = w;
    int resized_h = h;
    float scale_w = (float)box_w / w;
    float scale_h = (float)box_h / h;
    
    if (scale_w < scale_h) {
        resized_w = box_w;
        resized_h = resized_h * scale_w;
    } else {
        resized_h = h;
        resized_w = resized_w * scale_h;
    }
    //printf("\n\nletterbox gpu \n input w h : %d, %d\n resized w h: %d %d\n output w h: %d, %d\n", w, h, resized_w, resized_h, box_w, box_h);
	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(box_w,blockDim.x), iDivUp(box_h,blockDim.y));
    letterbox_image_kernel<<<gridDim, blockDim>>>(im_data, im_boxed, w, h, resized_w, resized_h, box_w, box_h); 
}
