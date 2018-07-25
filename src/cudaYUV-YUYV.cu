/*
 * http://github.com/dusty-nv/jetson-inference
 */

#include "cudaYUV.h"


inline __device__ __host__ float clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
}


/* From RGB to YUV

   Y = 0.299R + 0.587G + 0.114B
   U = 0.492 (B-Y)
   V = 0.877 (R-Y)

   It can also be represented as:

   Y =  0.299R + 0.587G + 0.114B
   U = -0.147R - 0.289G + 0.436B
   V =  0.615R - 0.515G - 0.100B

   From YUV to RGB

   R = Y + 1.140V
   G = Y - 0.395U - 0.581V
   B = Y + 2.032U
 */

struct __align__(8) uchar8
{
   uint8_t a0, a1, a2, a3, a4, a5, a6, a7;
};
static __host__ __device__ __forceinline__ uchar8 make_uchar8(uint8_t a0, uint8_t a1, uint8_t a2, uint8_t a3, uint8_t a4, uint8_t a5, uint8_t a6, uint8_t a7)
{
   uchar8 val = {a0, a1, a2, a3, a4, a5, a6, a7};
   return val;
}




//-----------------------------------------------------------------------------------
// YUYV/UYVY to RGBA
//-----------------------------------------------------------------------------------
template <bool formatUYVY>
__global__ void yuyvToRgba( uchar4* src, int srcAlignedWidth, uchar8* dst, int dstAlignedWidth, int width, int height )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= srcAlignedWidth || y >= height )
		return;

	const uchar4 macroPx = src[y * srcAlignedWidth + x];

	// Y0 is the brightness of pixel 0, Y1 the brightness of pixel 1.
	// U0 and V0 is the color of both pixels.
	// UYVY [ U0 | Y0 | V0 | Y1 ] 
	// YUYV [ Y0 | U0 | Y1 | V0 ]
	const float y0 = formatUYVY ? macroPx.y : macroPx.x;
	const float y1 = formatUYVY ? macroPx.w : macroPx.z; 
	const float u = (formatUYVY ? macroPx.x : macroPx.y) - 128.0f;
	const float v = (formatUYVY ? macroPx.z : macroPx.w) - 128.0f;

	const float4 px0 = make_float4( y0 + 1.4065f * v,
							  y0 - 0.3455f * u - 0.7169f * v,
							  y0 + 1.7790f * u, 255.0f );

	const float4 px1 = make_float4( y1 + 1.4065f * v,
							  y1 - 0.3455f * u - 0.7169f * v,
							  y1 + 1.7790f * u, 255.0f );

	dst[y * dstAlignedWidth + x] = make_uchar8( clamp(px0.x, 0.0f, 255.0f), 
									    clamp(px0.y, 0.0f, 255.0f),
									    clamp(px0.z, 0.0f, 255.0f),
									    clamp(px0.w, 0.0f, 255.0f),
									    clamp(px1.x, 0.0f, 255.0f),
									    clamp(px1.y, 0.0f, 255.0f),
									    clamp(px1.z, 0.0f, 255.0f),
									    clamp(px1.w, 0.0f, 255.0f) );
} 




template<bool formatUYVY>
cudaError_t launchYUYV( uchar2* input, size_t inputPitch, uchar4* output, size_t outputPitch, size_t width, size_t height)
{
	if( !input || !inputPitch || !output || !outputPitch || !width || !height )
		return cudaErrorInvalidValue;

	const dim3 block(8,8);
	const dim3 grid(iDivUp(width/2, block.x), iDivUp(height, block.y));

	const int srcAlignedWidth = inputPitch / sizeof(uchar4);	// normally would be uchar2, but we're doubling up pixels
	const int dstAlignedWidth = outputPitch / sizeof(uchar8);	// normally would be uchar4 ^^^

	//printf("yuyvToRgba %zu %zu %i %i %i %i %i\n", width, height, (int)formatUYVY, srcAlignedWidth, dstAlignedWidth, grid.x, grid.y);

	yuyvToRgba<formatUYVY><<<grid, block>>>((uchar4*)input, srcAlignedWidth, (uchar8*)output, dstAlignedWidth, width, height);

	return CUDA(cudaGetLastError());
}


cudaError_t cudaUYVYToRGBA( uchar2* input, uchar4* output, size_t width, size_t height )
{
	return cudaUYVYToRGBA(input, width * sizeof(uchar2), output, width * sizeof(uchar4), width, height);
}

cudaError_t cudaUYVYToRGBA( uchar2* input, size_t inputPitch, uchar4* output, size_t outputPitch, size_t width, size_t height )
{
	return launchYUYV<true>(input, inputPitch, output, outputPitch, width, height);
}





cudaError_t cudaYUYVToRGBA( uchar2* input, uchar4* output, size_t width, size_t height )
{
	return cudaYUYVToRGBA(input, width * sizeof(uchar2), output, width * sizeof(uchar4), width, height);
}

cudaError_t cudaYUYVToRGBA( uchar2* input, size_t inputPitch, uchar4* output, size_t outputPitch, size_t width, size_t height )
{
	return launchYUYV<false>(input, inputPitch, output, outputPitch, width, height);
}


//-----------------------------------------------------------------------------------
// YUYV/UYVY to grayscale
//-----------------------------------------------------------------------------------

template <bool formatUYVY>
__global__ void yuyvToGray( uchar4* src, int srcAlignedWidth, float2* dst, int dstAlignedWidth, int width, int height )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= srcAlignedWidth || y >= height )
		return;

	const uchar4 macroPx = src[y * srcAlignedWidth + x];

	const float y0 = formatUYVY ? macroPx.y : macroPx.x;
	const float y1 = formatUYVY ? macroPx.w : macroPx.z; 

	dst[y * dstAlignedWidth + x] = make_float2(y0/255.0f, y1/255.0f);
} 

template<bool formatUYVY>
cudaError_t launchGrayYUYV( uchar2* input, size_t inputPitch, float* output, size_t outputPitch, size_t width, size_t height)
{
	if( !input || !inputPitch || !output || !outputPitch || !width || !height )
		return cudaErrorInvalidValue;

	const dim3 block(8,8);
	const dim3 grid(iDivUp(width/2, block.x), iDivUp(height, block.y));

	const int srcAlignedWidth = inputPitch / sizeof(uchar4);	// normally would be uchar2, but we're doubling up pixels
	const int dstAlignedWidth = outputPitch / sizeof(float2);	// normally would be float ^^^

	yuyvToGray<formatUYVY><<<grid, block>>>((uchar4*)input, srcAlignedWidth, (float2*)output, dstAlignedWidth, width, height);

	return CUDA(cudaGetLastError());
}

cudaError_t cudaUYVYToGray( uchar2* input, float* output, size_t width, size_t height )
{
	return cudaUYVYToGray(input, width * sizeof(uchar2), output, width * sizeof(uint8_t), width, height);
}

cudaError_t cudaUYVYToGray( uchar2* input, size_t inputPitch, float* output, size_t outputPitch, size_t width, size_t height )
{
	return launchGrayYUYV<true>(input, inputPitch, output, outputPitch, width, height);
}

cudaError_t cudaYUYVToGray( uchar2* input, float* output, size_t width, size_t height )
{
	return cudaYUYVToGray(input, width * sizeof(uchar2), output, width * sizeof(float), width, height);
}

cudaError_t cudaYUYVToGray( uchar2* input, size_t inputPitch, float* output, size_t outputPitch, size_t width, size_t height )
{
	return launchGrayYUYV<false>(input, inputPitch, output, outputPitch, width, height);
}



//-----------------------------------------------------------------------------------
// YUYV/UYVY to V12
//-----------------------------------------------------------------------------------
template <bool formatUYVY>
__global__ void yuyvToYV12(unsigned char* src, unsigned char* dst, int width, int height )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;//width
	const int y = blockIdx.y * blockDim.y + threadIdx.y;//height

	if( x >= width || y >= height )
		return;
	
	const int pixel_index = y * width + x;
	
	const int frame_length = width * height;
	
	const int uv_length = frame_length >> 2;
	const int u_offset = frame_length;
	const int v_offset = frame_length + uv_length ;
	
	const int y_index = formatUYVY ? (pixel_index << 1 + 1) : pixel_index << 1;
	
	
	
	dst[pixel_index] = src[y_index];// Y
	if ( y%2 == 0)
	{
		const int uv_index = x/2 + y/2 * width / 2;
		if ( x % 2 == 0) {
			const int u_src = formatUYVY ? (pixel_index << 1) : (pixel_index << 1 + 1);
			dst[u_offset + uv_index] = src[u_src];
		} else {
			const int v_src = formatUYVY ? (pixel_index << 1) : (pixel_index << 1 + 1);
			dst[v_offset + uv_index] = src[v_src];
		}		
	}

} 

template<bool formatUYVY>
cudaError_t launchYUYV_YV12( unsigned char* input, unsigned char* output, size_t width, size_t height)
{
	if( !input || !output || !width || !height )
		return cudaErrorInvalidValue;

	const dim3 block(8,8);
	const dim3 grid(iDivUp(width, block.x), iDivUp(height, block.y));


	yuyvToYV12<formatUYVY><<<grid, block>>>((unsigned char*)input, (unsigned char*)output, width, height);

	return CUDA(cudaGetLastError());
}


cudaError_t cudaUYVYToYV12( unsigned char* input, unsigned char* output, size_t width, size_t height )
{
	return launchYUYV_YV12<true>(input, output, width, height);
}

cudaError_t cudaYUYVToYV12( unsigned char* input, unsigned char* output, size_t width, size_t height )
{
	return launchYUYV_YV12<false>(input, output, width, height);
}
/*
//-----------------------------------------------------------------------------------
// YUYV/UYVY to NV12
//-----------------------------------------------------------------------------------
template <bool formatUYVY>
__global__ void yuyvToNV12(unsigned char* src, unsigned char* dst, int width, int height )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;//width
	const int y = blockIdx.y * blockDim.y + threadIdx.y;//height

	if( x >= width || y >= height )
		return;
	
	const int pixel_index = y * width + x;
	
	const int frame_length = width * height;
	
	const int uv_length = frame_length >> 2;
	const int offset = frame_length;
	
	const int y_index = formatUYVY ? (pixel_index << 1 + 1) : pixel_index << 1;
	
	dst[pixel_index] = src[y_index];// Y
	if (y % 2 == 0) {
		const int uv_index = x/2 + y/2 * width / 2;
		if ( x % 2 == 0) {
			const int u_src = formatUYVY ? (pixel_index << 1) : (pixel_index << 1 + 1);
			dst[offset + 2 * uv_index] = src[u_src];
		} else {
			const int v_src = formatUYVY ? (pixel_index << 1) : (pixel_index << 1 + 1);
			dst[offset + 2 * uv_index + 1] = src[v_src];
		}
	}
} 

template<bool formatUYVY>
cudaError_t launchYUYV_NV12( unsigned char* input, unsigned char* output, size_t width, size_t height)
{
	if( !input || !output || !width || !height )
		return cudaErrorInvalidValue;

	const dim3 block(8,8);
	const dim3 grid(iDivUp(width, block.x), iDivUp(height, block.y));


	yuyvToNV12<formatUYVY><<<grid, block>>>((unsigned char*)input, (unsigned char*)output, width, height);

	return CUDA(cudaGetLastError());
}


cudaError_t cudaUYVYToNV12( unsigned char* input, unsigned char* output, size_t width, size_t height )
{
	return launchYUYV_NV12<true>(input, output, width, height);
}

cudaError_t cudaYUYVToNV12( unsigned char* input, unsigned char* output, size_t width, size_t height )
{
	return launchYUYV_NV12<false>(input, output, width, height);
}
*/
//-----------------------------------------------------------------------------------
// YUYV/UYVY to RGBA
//-----------------------------------------------------------------------------------
template <bool formatUYVY>
__global__ void yuyvToNv12( uchar4* src, int srcAlignedWidth, uchar2* dst, int dstAlignedWidth, int width, int height )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= srcAlignedWidth || y >= height )
		return;

	const uchar4 macroPx = src[y * srcAlignedWidth + x];

	const int size = width * height/ sizeof(uchar2);
	// Y0 is the brightness of pixel 0, Y1 the brightness of pixel 1.
	// U0 and V0 is the color of both pixels.
	// UYVY [ U0 | Y0 | V0 | Y1 ] 
	// YUYV [ Y0 | U0 | Y1 | V0 ]
	const float y0 = formatUYVY ? macroPx.y : macroPx.x;
	const float y1 = formatUYVY ? macroPx.w : macroPx.z; 
	dst[y * dstAlignedWidth + x] = make_uchar2(y0, y1);
	if(y%2 == 0)
	{
		const float u0 = (formatUYVY ? macroPx.x : macroPx.y);
		const float v1 = (formatUYVY ? macroPx.z : macroPx.w);
		dst[y/2 * dstAlignedWidth + x + size] = make_uchar2(u0, v1);
	}
} 

template<bool formatUYVY>
cudaError_t launchNv12YUYV( uchar2* input, size_t inputPitch, uint8_t* output, size_t outputPitch, size_t width, size_t height)
{
	if( !input || !inputPitch || !output || !outputPitch || !width || !height )
		return cudaErrorInvalidValue;

	const dim3 block(8,8);
	const dim3 grid(iDivUp(width/2, block.x), iDivUp(height, block.y));

	const int srcAlignedWidth = inputPitch / sizeof(uchar4);	// normally would be uchar2, but we're doubling up pixels
	const int dstAlignedWidth = outputPitch / sizeof(uchar2);	// normally would be uchar4 ^^^

	//printf("yuyvToRgba %zu %zu %i %i %i %i %i\n", width, height, (int)formatUYVY, srcAlignedWidth, dstAlignedWidth, grid.x, grid.y);

	yuyvToNv12<formatUYVY><<<grid, block>>>((uchar4*)input, srcAlignedWidth, (uchar2*)output, dstAlignedWidth, width, height);

	return CUDA(cudaGetLastError());
}

cudaError_t cudaUYVYToNV12( uchar2* input, uint8_t* output, size_t width, size_t height )
{
	return cudaUYVYToNV12(input, width * sizeof(uchar2), output, width, width, height);
}

cudaError_t cudaUYVYToNV12( uchar2* input, size_t inputPitch, uint8_t* output, size_t outputPitch, size_t width, size_t height )
{
	return launchNv12YUYV<true>(input, inputPitch, output, outputPitch, width, height);
}

cudaError_t cudaYUYVToNV12( uchar2* input, uint8_t* output, size_t width, size_t height )
{
	return cudaYUYVToNV12(input, width * sizeof(uchar2), output, width, width, height);
}

cudaError_t cudaYUYVToNV12( uchar2* input, size_t inputPitch, uint8_t* output, size_t outputPitch, size_t width, size_t height )
{
	return launchNv12YUYV<false>(input, inputPitch, output, outputPitch, width, height);
}



// ********* UYVY to RGB float (channel wise)**********

template <bool formatUYVY>
__global__ void yuyvToRgbf( uchar4* src, int srcAlignedWidth, float2* dst, int dstAlignedWidth, int width, int height )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= srcAlignedWidth || y >= height )
		return;

	const uchar4 macroPx = src[y * srcAlignedWidth + x];

	// Y0 is the brightness of pixel 0, Y1 the brightness of pixel 1.
	// U0 and V0 is the color of both pixels.
	// UYVY [ U0 | Y0 | V0 | Y1 ] 
	// YUYV [ Y0 | U0 | Y1 | V0 ]
	const float y0 = formatUYVY ? macroPx.y : macroPx.x;
	const float y1 = formatUYVY ? macroPx.w : macroPx.z; 
	const float u = (formatUYVY ? macroPx.x : macroPx.y) - 128.0f;
	const float v = (formatUYVY ? macroPx.z : macroPx.w) - 128.0f;

	
	float4 px0 = make_float4( (y0 + 1.4065f * v) / 255.f,
							  (y0 - 0.3455f * u - 0.7169f * v)/255.f,
							  (y0 + 1.7790f * u)/255.f, 1.0f );
		
	
	float4 px1 = make_float4( (y1 + 1.4065f * v)/255.f,
							  (y1 - 0.3455f * u - 0.7169f * v)/255.f,
							  (y1 + 1.7790f * u)/255.f, 1.0f );
		
	const int offset = dstAlignedWidth * height;
	
	dst[y*dstAlignedWidth + x] = make_float2(clamp(px0.x, 0.0f, 1.0f), clamp(px1.x, 0.0f, 1.0f));
	dst[y*dstAlignedWidth + x + offset] = make_float2(clamp(px0.y, 0.0f, 1.0f), clamp(px1.y, 0.0f, 1.0f));
	dst[y*dstAlignedWidth + x + 2 * offset] = make_float2(clamp(px0.z, 0.0f, 1.0f), clamp(px1.z, 0.0f, 1.0f));
	
} 

template<bool formatUYVY>
cudaError_t launchYUYV_RGBf( uchar2* input, size_t inputPitch, float* output, size_t outputPitch, size_t width, size_t height)
{
	if( !input || !inputPitch || !output || !outputPitch || !width || !height )
		return cudaErrorInvalidValue;

	const dim3 block(8,8);
	const dim3 grid(iDivUp(width/2, block.x), iDivUp(height, block.y));

	const int srcAlignedWidth = inputPitch / sizeof(uchar4);	// normally would be uchar2, but we're doubling up pixels
	const int dstAlignedWidth = outputPitch / sizeof(float2);	// normally would be uchar4 ^^^

	//printf("yuyvToRgba %zu %zu %i %i %i %i %i\n", width, height, (int)formatUYVY, srcAlignedWidth, dstAlignedWidth, grid.x, grid.y);

	yuyvToRgbf<formatUYVY><<<grid, block>>>((uchar4*)input, srcAlignedWidth, (float2*)output, dstAlignedWidth, width, height);

	return CUDA(cudaGetLastError());
}



cudaError_t cudaUYVYToRGBf( uchar2* input, size_t inputPitch, float* output, size_t outputPitch, size_t width, size_t height )
{
	return launchYUYV_RGBf<true>(input, inputPitch, output, outputPitch, width, height);
}

cudaError_t cudaUYVYToRGBf( uchar2* input, float* output, size_t width, size_t height )
{
	return cudaUYVYToRGBf(input, width * sizeof(uchar2), output, width * sizeof(float), width, height);
}



cudaError_t cudaYUYVToRGBf( uchar2* input, size_t inputPitch, float* output, size_t outputPitch, size_t width, size_t height )
{
	return launchYUYV_RGBf<false>(input, inputPitch, output, outputPitch, width, height);
}

cudaError_t cudaYUYVToRGBf( uchar2* input, float* output, size_t width, size_t height )
{
	return cudaYUYVToRGBf(input, width * sizeof(uchar2), output, width * sizeof(float), width, height);
}
