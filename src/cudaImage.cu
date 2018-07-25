/*
 * inference-101
 */
//extern "C" {
#include "cudaImage.h"
//}

__device__ uchar3 get_pixel(float* data, uint32_t w, uint32_t h, int x, int y)
{
	const unsigned char r = data[y*w + x]*255;
	const unsigned char g = data[y*w + x + h*w]*255;
	const unsigned char b = data[y*w + x + h*w*2]*255;

    return make_uchar3(r, g, b);
}

__device__ float get_pixel_gpu(float* data, uint32_t w, uint32_t h, uint32_t c, uint32_t imw, uint32_t imh, uint32_t imc)
{
	if(w < imw && h < imh && c < imc)
	{
		return data[c*imw*imh + h*imw + w];
	}
	else return 0;
}

__device__ void set_pixel_gpu(float* data, float val, uint32_t w, uint32_t h, uint32_t c, uint32_t imw, uint32_t imh, uint32_t imc)
{
	if(w >= imw || h >= imh || c >= imc || w < 0|| h < 0 || c < 0) return;
	data[c*imw*imh + h*imw + w] = val;

}

__global__ void resize_image_width(float* srcImage, float* dstImage, uint32_t srcw, uint32_t dstw, uint32_t srch, uint32_t srcc, float w_scale)
{
	int x, y;

    x = (blockIdx.x * blockDim.x) + threadIdx.x;
    y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
    //~ Rpixel = y * srcw + x;
    //~ Gpixel = srcw*srch + y * srcw + x;
    //~ Bpixel = 2*srcw*srch + y * srcw + x;
    if(x >= dstw || y >= srch) return ;
	for(int c = 0; c < srcc; c++)
	{
		float val = 0;
		if((x == dstw - 1) || (srcw == 1))
		{
			val = get_pixel_gpu(srcImage, srcw - 1 , y, c, srcw, srch, srcc);
		}
		else
		{
			float sx = x*w_scale;
			int ix = (int) sx;
			float dx = sx - ix;
			val = (1 - dx)*get_pixel_gpu(srcImage, ix , y, c, srcw, srch, srcc)
					+ dx*get_pixel_gpu(srcImage, ix+1 , y, c, srcw, srch, srcc);
			
		}
		set_pixel_gpu(dstImage, val, x , y, c, dstw, srch, srcc);
	}
}

cudaError_t cudaImageResizeWidth(float* srcImage, float* dstImage, uint32_t srcw, uint32_t dstw, uint32_t srch, uint32_t srcc, float w_scale) 
{
	if( !srcImage || !dstImage )
	return cudaErrorInvalidDevicePointer;
	const dim3 blockDim(8,8,1);
	const dim3 gridDim(iDivUp(dstw,blockDim.x), iDivUp(srch,blockDim.y), 1);
	resize_image_width<<<gridDim , blockDim>>>(srcImage, dstImage, srcw, dstw, srch, srcc, w_scale);
}

__global__ void resize_image_height(float* srcImage, float* dstImage, uint32_t srch, uint32_t dsth, uint32_t srcw, uint32_t srcc, float h_scale)
{
	int x, y;

    x = (blockIdx.x * blockDim.x) + threadIdx.x;
    y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
    if(y >= dsth || x >= srcw) return ;
	for(int c = 0; c < srcc; c++)
	{
		float sy = y*h_scale;
		int iy = (int) sy;
		float dy = sy - iy;
		float val = (1-dy)*get_pixel_gpu(srcImage, x, iy, c, srcw, srch, srcc);
		set_pixel_gpu(dstImage, val, x , y, c, srcw, dsth, srcc);
		if(y != dsth-1 && srch != 1)
		{
			float add_val = dy*get_pixel_gpu(srcImage, x, iy+1, c, srcw, srch, srcc);
			val += add_val;
			set_pixel_gpu(dstImage, val, x , y, c, srcw, dsth, srcc);
		}
	}
}

cudaError_t cudaImageResizeHeight(float* srcImage, float* dstImage, uint32_t srch, uint32_t dsth, uint32_t srcw, uint32_t srcc, float h_scale) 
{
	if( !srcImage || !dstImage )
	return cudaErrorInvalidDevicePointer;
	const dim3 blockDim(8,8,1);
	const dim3 gridDim(iDivUp(srcw,blockDim.x), iDivUp(dsth,blockDim.y), 1);
	resize_image_height<<<gridDim , blockDim>>>(srcImage, dstImage, srch, dsth, srcw, srcc, h_scale);
}

void resize_image_gpu(float* srcImage, float* dstImage, float *tmpImage, uint32_t srcw, uint32_t srch, uint32_t dstw, uint32_t dsth)
{
	float w_scale = (float)(srcw - 1)/(dstw - 1);
	float h_scale = (float)(srch - 1)/(dsth - 1);
	//void *tmpImage = NULL;
	//checkCudaErrors(cudaMalloc((void **)&tmpImage, dstw * dsth * 3 * sizeof(float)));
	CUDA_FAILED(cudaImageResizeWidth(srcImage, tmpImage, srcw, dstw, srch, 3, w_scale));
	CUDA_FAILED(cudaImageResizeHeight(tmpImage, dstImage, srch, dsth, dstw, 3, h_scale));
	
}

__global__ void imageRGBToMatData(float* srcImage,
                           uchar3* dstImage, uint32_t step,
                           uint32_t width, uint32_t height)
{
    int x, y, pixel;

    x = (blockIdx.x * blockDim.x) + threadIdx.x;
    y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
    pixel = y * step + x;

    if (x >= step)
        return; 

    if (y >= height)
        return;

//	printf("cuda thread %i %i  %i %i pixel %i \n", x, y, width, height, pixel);
		
	// const float3  rgb = get_pixel(srcImage, width, height, x, y);

	const uchar3 px = get_pixel(srcImage, width, height, x, y);
	
	dstImage[pixel] = px;
}

// cudaResizeRGBA
cudaError_t cudaImageRGBToMatData( float* srcDev, uchar3* destDev, size_t step, size_t width, size_t height)
{
	if( !srcDev || !destDev )
		return cudaErrorInvalidDevicePointer;

	const dim3 blockDim(8,8,1);
	const dim3 gridDim(iDivUp(step,blockDim.x), iDivUp(height,blockDim.y), 1);

	imageRGBToMatData<<<gridDim, blockDim>>>( srcDev, destDev, step, width, height );
	
	return CUDA(cudaGetLastError());

}

__global__ void letterbox_kernel( const float *input,
                                  float *output,
                                  const int width_in,
                                  const int height_in,
                                  const int width_resized,
                                  const int height_resized,                                  
                                  const int width_out,
                                  const int height_out)
{
    const float scale = (float)(height_in * 1.f/height_resized);
    const unsigned int xIndex = blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned int yIndex = blockIdx.y*blockDim.y+threadIdx.y;
    const unsigned int zIndex = blockIdx.z*blockDim.z+threadIdx.z;
    
    if(xIndex >= width_out || yIndex >= height_out || zIndex >= 3)
        return;
    
    unsigned int tid = yIndex * width_out + xIndex + zIndex * width_out * height_out;
    //~ unsigned int offset_out = width_out * height_out;
    //~ unsigned int offset_resized = width_resized * height_resized;
    //~ unsigned int offset_in = width_in * height_in;

    unsigned int dx = (width_out - width_resized) >> 1;
    unsigned int dy = (height_out - height_resized) >> 1;
    
    if (xIndex <= dx || xIndex >= (width_out - dx) || yIndex <= dy || yIndex >= (height_out - dy))
    {
	    output[tid] = 0.5f;
	    //~ output[tid + offset_out] = 0.5f;
	    //~ output[tid + offset_out << 1] = 0.5f;
    } else {
	    unsigned int x = xIndex - dx;
	    unsigned int y = yIndex - dy;
	    float inXindex = (float)(x * 1.f * scale) + 0.5f;
	    float inYindex = (float)(y * 1.f * scale) + 0.5f;
    
	    int intX = (int)(inXindex + 0.5f);
	    int intY = (int)(inYindex + 0.5f);
    
	    float a = inXindex - intX + 0.5f;
	    float b = inYindex - intY + 0.5f;
		
		//~ unsigned int p1 = intY * width_in + intX;
	    //~ unsigned int p2 = intY * width_in + intX + 1;
	    //~ unsigned int p3 = (intY+1) * width_in + intX;
	    //~ unsigned int p4 = (intY+1) * width_in + intX + 1;
	    
	    unsigned int p1 = intY * width_in + intX + zIndex * width_in * height_in;
	    unsigned int p2 = intY * width_in + intX + 1 + zIndex * width_in * height_in;
	    unsigned int p3 = (intY+1) * width_in + intX + zIndex * width_in * height_in;
	    unsigned int p4 = (intY+1) * width_in + intX + 1 + zIndex * width_in * height_in;
		
		if(p1 >= (height_in * width_in*3) || p2 >= (height_in * width_in*3) || p3 >= (height_in * width_in*3) || p4 >= (height_in * width_in*3) )
		{
			//printf("intY: %d, intX: %d\n", intY, intX);
			return;
		}
		
		//~ printf("X:%d, Y:%d, Z:%d", xIndex, yIndex, zIndex);
		
        output[tid] = (1-a) * (1-b)* input[p1] + 
						  a * (1-b)* input[p2] + 
					  (1-a) *    b * input[p3] + 
						  a *    b * input[p4];
	    //~ output[tid + (width_out * height_out)] = (1-a) * (1-b) * input[p1 + width_in * height_in] + 
													 //~ a * (1-b) * input[p2 + width_in * height_in] + 
												 //~ (1-a) *    b  * input[p3 + width_in * height_in] +
													 //~ a *    b  * input[p4 + width_in * height_in];
	    //~ output[tid + (width_out * height_out) << 1] = (1-a) * (1-b) * input[p1 + (width_in * height_in) << 1] + 
	                                        //~ a * (1-b) * input[p2 + (width_in * height_in) << 1] + 
	                                    //~ (1-a) *    b  * input[p3 + (width_in * height_in) << 1] + 
	                                        //~ a *    b  * input[p4 + (width_in * height_in) << 1];
                //~ if(x < 10 && y < 10)
	              //~ printf("\n offset:(%d %d) out_index: (%d %d) in_index: (%d %d) scale:(%f %f) \n out(rgb): %f %f %f\n", dx, dy, xIndex, yIndex, intX, intY, scale, scale, p1, p2, p3, p4, output[tid],output[tid+offset_out], output[tid+offset_out*2]);
    }
}



cudaError_t cudaLetterboxImage(const float *im_data, const int w, const int h, float *output, const int net_w, const int net_h)
{
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    
    if( !im_data || !output )
		return cudaErrorInvalidDevicePointer;
		
	
    int resized_w = w;
    int resized_h = h;
        
    if (((float)net_w/w) < ((float)net_h/h)) {
        resized_w = net_w;
        resized_h = (h * net_w) / w;
    } else {
        resized_h = net_h;
        resized_w = (w * net_h) / h;
    }
    //printf("\n\nletterbox gpu \n inputs w h : %d, %d\n resized w h: %d %d\n output w h: %d, %d\n", w, h, resized_w, resized_h, net_w, net_h);
	// launch kernel
	const dim3 blockDim(8, 8, 3);
	const dim3 gridDim(iDivUp(net_w,blockDim.x), iDivUp(net_h,blockDim.y), 3);
    letterbox_kernel<<<gridDim, blockDim>>>(im_data, output, w, h, resized_w, resized_h, net_w, net_h);
    
    return CUDA(cudaGetLastError());
}



__global__ void crop_kernel( const float *input,
                                  const int w_in,
                                  const int h_in,
                                  float *output,
                                  const int dx,
                                  const int dy,                                  
                                  const int w_out,
                                  const int h_out,
                                  const int channel)
{
    const unsigned int x = blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y+threadIdx.y;
    const unsigned int z = blockIdx.z*blockDim.z+threadIdx.z;
    
    if(x >= w_out || y >= h_out || z >= channel)
        return;
    
    int tid = y * w_out + x + z * w_out * h_out;
    
    int x_src = x + dx;
    int y_src = y + dy;
    
    
    if(x_src >= w_in || y_src >= h_in)
        return;

    int idx_src = y_src * w_in + x_src + z * w_in* h_in;
    
    output[tid] = input[idx_src];
    //output[tid + w_out*h_out] = input[idx_src + w_in*h_in];
    //output[tid + 2 * w_out * h_out] = input[idx_src + 2 * w_in*h_in];
}


cudaError_t cudaCropImage(const float *im_data, const int w, const int h, float *output, int dx, int dy, int new_w, int new_h, const int channel)
{
	if( !im_data || !output )
		return cudaErrorInvalidDevicePointer;
		
	printf("crop image (%d, %d) -->(%d, %d, %d, %d)\n", w, h, dx, dy, new_w, new_h);
	// launch kernel
	const dim3 blockDim(8, 8, channel);
	const dim3 gridDim(iDivUp(new_w,blockDim.x), iDivUp(new_h,blockDim.y), channel);
    crop_kernel<<<gridDim, blockDim>>>(im_data, w, h, output, dx, dy, new_w, new_h, channel);
    
    return CUDA(cudaGetLastError());
    
}
