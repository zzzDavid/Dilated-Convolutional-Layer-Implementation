#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "activations.h"

#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"
}

__global__ void binarize_kernel(float *x, int n, float *binary)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    binary[i] = (x[i] >= 0) ? 1 : -1;
}

void binarize_gpu(float *x, int n, float *binary)
{
    binarize_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, binary);
    check_error(cudaPeekAtLastError());
}

__global__ void binarize_input_kernel(float *input, int n, int size, float *binary)
{
    int s = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (s >= size) return;
    int i = 0;
    float mean = 0;
    for(i = 0; i < n; ++i){
        mean += abs(input[i*size + s]);
    }
    mean = mean / n;
    for(i = 0; i < n; ++i){
        binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
    }
}

void binarize_input_gpu(float *input, int n, int size, float *binary)
{
    binarize_input_kernel<<<cuda_gridsize(size), BLOCK>>>(input, n, size, binary);
    check_error(cudaPeekAtLastError());
}


__global__ void binarize_weights_kernel(float *weights, int n, int size, float *binary)
{
    int f = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (f >= n) return;
    int i = 0;
    float mean = 0;
    for(i = 0; i < size; ++i){
        mean += abs(weights[f*size + i]);
    }
    mean = mean / size;
    for(i = 0; i < size; ++i){
        binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        //binary[f*size + i] = weights[f*size + i];
    }
}

void binarize_weights_gpu(float *weights, int n, int size, float *binary)
{
    binarize_weights_kernel<<<cuda_gridsize(n), BLOCK>>>(weights, n, size, binary);
    check_error(cudaPeekAtLastError());
}
__global__
void forward_depthwise_convolution_kernel_add_bias(const int n, const float* __restrict__ data_im,
        const int height, const int width, const int ksize,
        const int pad,
        const int stride,
        const int height_col, const int width_col,
        float * __restrict__ im_col,
        const float* __restrict__ weights,
        const float* __restrict__ biases,
        ACTIVATION activation) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index < n){
		int h_index = index / width_col;
		int channel_in = h_index / height_col;
		int w_out = index - h_index * width_col;
		int h_out = h_index - channel_in * height_col;
		int h_in = h_out * stride - pad;
		int w_in = w_out * stride - pad;
		const float* data_im_ptr = data_im;
		data_im_ptr += (channel_in * height + h_in) * width + w_in;

		int kernel_group_offset = channel_in * ksize * ksize;

		float acc = 0;
#if 0
#pragma unroll
		for (int i = 0; i < 3; ++i) {
#pragma unroll
			for (int j = 0; j < 3; ++j) {
				int h = h_in + i;
				int w = w_in + j;

				bool tmp = (h >= 0 && w >= 0 && h < height && w < width);
				int input_loc = i * width + j;
				int kernel_loc = kernel_group_offset + i * ksize + j;
				float im_data = (tmp ? data_im_ptr[input_loc] : 0);
				acc += im_data * weights[kernel_loc];
			}
		}
#endif
// double buffer
#if 1
		int h[2], w[2], input_loc[2], kernel_loc[2];
		bool tmp[2]; float im_data[2], weight_data[2];

		h[0] = h_in;
		w[0] = w_in;
		input_loc[0] = 0;
		kernel_loc[0] = kernel_group_offset;
		tmp[0] = (h[0] >= 0 && w[0] >= 0 && h[0] < height && w[0] < width);
		im_data[0] = (tmp ? data_im_ptr[input_loc[0]] : 0);
		weight_data[0] = weights[kernel_loc[0]];

		const int idx_i[8] = {0, 0, 1, 1, 1, 2, 2, 2};
		const int idx_j[8] = {1, 2, 0, 1, 2, 0, 1, 2};


#pragma unroll
		for(int i = 0; i < 8; i++) {
			int next_idx = (i+1) & 1;
			int curr_idx = i&1;
			h[next_idx] = h_in + idx_i[i];
			w[next_idx] = w_in + idx_j[i];
			input_loc[next_idx] = idx_i[i] * width + idx_j[i];
			kernel_loc[next_idx] = kernel_group_offset + idx_i[i] * ksize + idx_j[i];
			tmp[next_idx] = (h[next_idx] >= 0 && w[next_idx] >= 0
					&& h[next_idx] < height && w[next_idx] < width);
			weight_data[next_idx] = weights[kernel_loc[next_idx]];
			im_data[next_idx] = (tmp[next_idx] ? data_im_ptr[input_loc[next_idx]] : 0);

			acc += im_data[curr_idx] * weight_data[curr_idx];

		}

		acc += im_data[0] * weight_data[0];

#endif



		acc+=biases[channel_in];
		switch(activation){
			case LINEAR:
			    acc=acc;break;
			case LOGISTIC:
			    acc= 1./(1. + exp(-acc));break;
			case LOGGY:
			    acc=  2./(1. + exp(-acc)) - 1;break;
			case RELU:
			    acc= acc*(acc>0);break;
			case ELU:
			    acc= (acc >= 0)*acc + (acc < 0)*(exp(acc)-1);break;
			case RELIE:
			    acc= (acc>0) ? acc : .01*acc;break;
			case RAMP:
			    acc= acc*(acc>0)+.1*acc;break;
			case LEAKY:
			    acc= (acc>0) ? acc : .1*acc;break;
			case TANH:
			    acc= (2/(1 + exp(-2*acc)) - 1);break;
			case PLSE:{
			    if(acc < -4) acc= .01 * (acc + 4);
			        if(acc > 4)  acc= .01 * (acc - 4) + 1;
			            acc= .125*acc + .5;
			}break;

			case STAIR:{
			   int n = floor(acc);
			       if (n%2 == 0) acc= floor(acc/2.);
			       else acc= (acc - n) + floor(acc/2.);
			}break;
			case HARDTAN:{
			    if (acc > -1 && acc < 1) acc= 1;
			    else acc=0;
			}break;
		       /* case LHTAN:
			    acc= lhtan_activate_kernel(acc);*/
		    }
		im_col[index] = acc;
	}
}


__global__
void forward_depthwise_convolution_kernel_add_bias_relu(const int n, const float* __restrict__ data_im,
        const int height, const int width, const int ksize,
        const int pad,
        const int stride,
        const int height_col,
        const int width_col,
        const unsigned magic_height,
        const unsigned magic_width,
        const int magic_height_shift,
        const int magic_width_shift,
        float * __restrict__ im_col,
        const float* __restrict__ weights,
        const float* __restrict__ biases) {

	unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index < n){

		//int h_index = index / width_col;
		//int channel_in = h_index / height_col;
		//int w_out = index % width_col;
		//int h_out = h_index % height_col;
		// below code perform the above code function
		// using magic number to avoid division and modulo
		int h_index = __umulhi(index, magic_width);
		h_index = h_index >> magic_width_shift;
		int channel_in = __umulhi(h_index, magic_height);
		channel_in = channel_in >> magic_height_shift;
		int w_out = index - h_index * width_col;
		int h_out = h_index - channel_in * height_col;


		int h_in = h_out * stride - pad;
		int w_in = w_out * stride - pad;
		const float* data_im_ptr = data_im;
		data_im_ptr += (channel_in * height + h_in) * width + w_in;

		int kernel_group_offset = channel_in * ksize * ksize;

		float acc = 0;
#if 0
#pragma unroll
		for (int i = 0; i < 3; ++i) {
#pragma unroll
			for (int j = 0; j < 3; ++j) {
				int h = h_in + i;
				int w = w_in + j;

				bool tmp = (h >= 0 && w >= 0 && h < height && w < width);
				int input_loc = i * width + j;
				int kernel_loc = kernel_group_offset + i * ksize + j;
				float im_data = (tmp ? data_im_ptr[input_loc] : 0);
				acc += im_data * weights[kernel_loc];
			}
		}
#endif
// double buffer
#if 1
		int h[2], w[2], input_loc[2], kernel_loc[2];
		bool tmp[2]; float im_data[2], weight_data[2];

		h[0] = h_in;
		w[0] = w_in;
		input_loc[0] = 0;
		kernel_loc[0] = kernel_group_offset;
		tmp[0] = (h[0] >= 0 && w[0] >= 0 && h[0] < height && w[0] < width);
		im_data[0] = (tmp[0] ? data_im_ptr[input_loc[0]] : 0);
		weight_data[0] = weights[kernel_loc[0]];

		const int idx_i[8] = {0, 0, 1, 1, 1, 2, 2, 2};
		const int idx_j[8] = {1, 2, 0, 1, 2, 0, 1, 2};


#pragma unroll
		for(int i = 0; i < 8; i++) {
			int next_idx = (i+1) & 1;
			int curr_idx = i&1;
			h[next_idx] = h_in + idx_i[i];
			w[next_idx] = w_in + idx_j[i];
			input_loc[next_idx] = idx_i[i] * width + idx_j[i];
			kernel_loc[next_idx] = kernel_group_offset + idx_i[i] * ksize + idx_j[i];
			tmp[next_idx] = (h[next_idx] >= 0 && w[next_idx] >= 0
					&& h[next_idx] < height && w[next_idx] < width);
			weight_data[next_idx] = weights[kernel_loc[next_idx]];
			im_data[next_idx] = (tmp[next_idx] ? data_im_ptr[input_loc[next_idx]] : 0);

			acc += im_data[curr_idx] * weight_data[curr_idx];

		}

		acc += im_data[0] * weight_data[0];

#endif



		acc+=biases[channel_in];
		acc = (acc>0) * acc;
		im_col[index] = acc;
	}
}


typedef struct{
	unsigned magic_number;
	int shift;
} MagicNumberAndShift;

MagicNumberAndShift calculate_magic_number(int denominator) {
	int d = denominator;
	int shift = 0;
	unsigned nc = 0x7FFFFFFF / d * d - 1;
	long long unsigned twoP = 0xFFFFFFFF;
	unsigned m = 0;
	for (int j = 0; j < 32; j++) {
		long long unsigned comp = ((long long unsigned)nc) * (d - twoP % d);
		if(twoP > comp) {
			m = (twoP + d - twoP%d)/d;
		    break;
		}
		else {
		    shift ++;
		    twoP = twoP << 1;
		    twoP |= 0x1;
		}

	}
	MagicNumberAndShift ret;
	ret.magic_number = m;
	ret.shift = shift;
	return ret;
}


void forward_convolutional_layer_gpu(convolutional_layer l, network net)
{
	// comment by Tiandong Wang: why set output to zero and set BETA
	// in gemm or cudnnConvFwd to 1? this really waste of time!
	//fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    if(l.binary){
        binarize_weights_gpu(l.weights_gpu, l.n, l.c*l.size*l.size, l.binary_weights_gpu);
        swap_binary(&l);
    }

    if(l.xnor){
        binarize_weights_gpu(l.weights_gpu, l.n, l.c*l.size*l.size, l.binary_weights_gpu);
        swap_binary(&l);
        binarize_gpu(net.input_gpu, l.c*l.h*l.w*l.batch, l.binary_input_gpu);
        net.input_gpu = l.binary_input_gpu;
    }

#ifdef CUDNN
    float one = 1;
    float zero = 0.0;
    //int weight_offset = k / l.groups;
    //int col_offset = l.size * l.size * l.out_h * l.out_w;
    //int output_offset = m * l.out_w * l.out_h / l.groups;
    //int j;
    cudnnConvolutionForward(cudnn_handle(),
                &one,
                l.srcTensorDesc,
                net.input_gpu,
                l.weightDesc,
                l.weights_gpu,
                l.convDesc,
                l.fw_algo,
                net.workspace,
                l.workspace_size,
                &zero,
                l.dstTensorDesc,
                l.output_gpu);

#else
    int i;
    int m = l.n;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_w*l.out_h;

    int weight_offset = m * k / l.groups;
    int col_offset = k * n;
    int output_offset = m * n / l. groups;

    for(i = 0; i < l.batch; ++i){
        float * a = l.weights_gpu;
        float * b = net.workspace;
        float * c = l.output_gpu + i*m*n;

        if(m == l.groups && l.size == 3 && !l.batch_normalize) {
            int height_col = (l.h + 2 * l.pad - l.size) / l.stride + 1;
            int width_col = (l.w + 2 * l.pad - l.size) / l.stride + 1;
            int num_kernels = l.c * height_col * width_col;
            if(l.activation == RELU) {
            	MagicNumberAndShift magic_height = calculate_magic_number(height_col);
            	MagicNumberAndShift magic_width = calculate_magic_number(width_col);

				forward_depthwise_convolution_kernel_add_bias_relu<<<(num_kernels+BLOCK-1)/BLOCK,
					BLOCK>>>(num_kernels, net.input_gpu + i*l.c*l.h*l.w,
						l.h, l.w, l.size, l.pad, l.stride,
						height_col, width_col,

						magic_height.magic_number,
						magic_width.magic_number,
						magic_height.shift,
						magic_width.shift,

						l.output_gpu + i*m*n,
						l.weights_gpu,
						l.biases_gpu);
            }
            else {
            	forward_depthwise_convolution_kernel_add_bias<<<(num_kernels+BLOCK-1)/BLOCK,
					BLOCK>>>(num_kernels, net.input_gpu + i*l.c*l.h*l.w,
						l.h, l.w, l.size, l.pad, l.stride,
						height_col, width_col,
						l.output_gpu + i*m*n,
						l.weights_gpu,
						l.biases_gpu,
						l.activation);
            }
        }
        else if(l.size == 1 && l.groups == 1) { //pw conv
        	b = net.input_gpu + i*l.c*l.h*l.w;
    	    group_gemm_gpu(0,0,m / l.groups,n,k,1.,a, k, weight_offset, b, n, col_offset, 0.0, c, n, output_offset, l.groups);
    	    if (l.batch_normalize) {
    	    	forward_batchnorm_layer_gpu(l, net);
        	    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
    	    }
    	    else if(l.activation == RELU) {
    	    	activate_array_relu_with_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
    	    }
    	    else {
				add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
				activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
    	    }

        }
        else {
			im2col_gpu(net.input_gpu + i*l.c*l.h*l.w, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, net.workspace);
			group_gemm_gpu(0,0,m / l.groups,n,k,1.,a, k, weight_offset, b, n, col_offset, 0.0, c, n, output_offset, l.groups);
			if (l.batch_normalize) {
				forward_batchnorm_layer_gpu(l, net);
			} else {
				add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
			}
			activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
        }
    }
#endif
    //if(l.dot > 0) dot_error_gpu(l);
    if(l.binary || l.xnor) swap_binary(&l);
}

__global__ void smooth_kernel(float *x, int n, int w, int h, int c, int size, float rate, float *delta)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int j = id % w;
    id /= w;
    int i = id % h;
    id /= h;
    int k = id % c;
    id /= c;
    int b = id;

    int w_offset = -(size/2.);
    int h_offset = -(size/2.);

    int out_index = j + w*(i + h*(k + c*b));
    int l, m;
    for(l = 0; l < size; ++l){
        for(m = 0; m < size; ++m){
            int cur_h = h_offset + i + l;
            int cur_w = w_offset + j + m;
            int index = cur_w + w*(cur_h + h*(k + b*c));
            int valid = (cur_h >= 0 && cur_h < h &&
                    cur_w >= 0 && cur_w < w);
            delta[out_index] += valid ? rate*(x[index] - x[out_index]) : 0;
        }
    }
}

extern "C" void smooth_layer(layer l, int size, float rate)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.out_c;

    size_t n = h*w*c*l.batch;

    smooth_kernel<<<cuda_gridsize(n), BLOCK>>>(l.output_gpu, n, l.w, l.h, l.c, size, rate, l.delta_gpu);
    check_error(cudaPeekAtLastError());
}

void backward_convolutional_layer_gpu(convolutional_layer l, network net)
{
    if(l.smooth){
        smooth_layer(l, 5, l.smooth);
    }
    constrain_gpu(l.outputs*l.batch, 1, l.delta_gpu, 1);
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);


    if(l.batch_normalize){
        backward_batchnorm_layer_gpu(l, net);
    } else {
        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);
    }
    float *original_input = net.input_gpu;

    if(l.xnor) net.input_gpu = l.binary_input_gpu;
#ifdef CUDNN
    float one = 1;
    cudnnConvolutionBackwardFilter(cudnn_handle(),
            &one,
            l.srcTensorDesc,
            net.input_gpu,
            l.ddstTensorDesc,
            l.delta_gpu,
            l.convDesc,
            l.bf_algo,
            net.workspace,
            l.workspace_size,
            &one,
            l.dweightDesc,
            l.weight_updates_gpu);

    if(net.delta_gpu){
        if(l.binary || l.xnor) swap_binary(&l);
        cudnnConvolutionBackwardData(cudnn_handle(),
                &one,
                l.weightDesc,
                l.weights_gpu,
                l.ddstTensorDesc,
                l.delta_gpu,
                l.convDesc,
                l.bd_algo,
                net.workspace,
                l.workspace_size,
                &one,
                l.dsrcTensorDesc,
                net.delta_gpu);
        if(l.binary || l.xnor) swap_binary(&l);
        if(l.xnor) gradient_array_gpu(original_input, l.batch*l.c*l.h*l.w, HARDTAN, net.delta_gpu);
    }

#else
    int m = l.n;
    int n = l.size*l.size*l.c/l.groups;
    int k = l.out_w*l.out_h;

    int weight_offset = m * n / l.groups;
    int col_offset = k * n;
    int output_offset = m * k / l. groups;

    int i;

    //printf("layer m,n,...,output_offset: %d %d %d %d %d %d \n", m, n, k, weight_offset, col_offset, output_offset);
    for(i = 0; i < l.batch; ++i){
        float * a = l.delta_gpu + i*m*k;
        float * b = net.workspace;
        float * c = l.weight_updates_gpu;

        im2col_gpu(net.input_gpu + i*l.c*l.h*l.w, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, net.workspace);
        //gemm_gpu(0,1,m,n,k,1,a + i*m*k,k,b,k,1,c,n);
        group_gemm_gpu(0, 1, m / l.groups, n,k, 1, a, k, output_offset,b, k, col_offset,1, c, n, weight_offset, l.groups);

/*
        for (g=0; g<l.groups; g++) {
            //printf("groups: %d %d %d %d %d %d \n", m / l.groups, n, k, output_offset, col_offset, weight_offset);
            gemm_gpu(0, 1,
                 m / l.groups, n,
                 k, 1, a + output_offset * g, k,
                 b + col_offset * g, k,
                 1, c + weight_offset * g, n);
        }
*/


        if(net.delta_gpu){
            //printf("net.delta_gpu exists!");
            if(l.binary || l.xnor) swap_binary(&l);
            a = l.weights_gpu;
            b = l.delta_gpu + i*m*k;
            c = net.workspace;

            //gemm_gpu(1,0,n,k,m,1,a,n,b + i*k*m,k,0,c,k);
            group_gemm_gpu(1, 0, n, k, m/l.groups, 1, a, n,weight_offset, b, k,output_offset, 0, c, k, col_offset, l.groups);
/*
            for(g=0; g<l.groups; g++) {
                gemm_gpu(1, 0, n, k, m/l.groups, 1, a + weight_offset * g, n, b + output_offset * g, k, 0, c + col_offset * g, k);
            }
*/
            col2im_gpu(net.workspace, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, net.delta_gpu + i*l.c*l.h*l.w);
            if(l.binary || l.xnor) {
                swap_binary(&l);
            }
            if(l.xnor) gradient_array_gpu(original_input + i*l.c*l.h*l.w, l.c*l.h*l.w, HARDTAN, net.delta_gpu + i*l.c*l.h*l.w);
        }
    }
#endif
}

void pull_convolutional_layer(convolutional_layer layer)
{
    cuda_pull_array(layer.weights_gpu, layer.weights, layer.c*layer.n*layer.size*layer.size/layer.groups);
    cuda_pull_array(layer.biases_gpu, layer.biases, layer.n);
    cuda_pull_array(layer.weight_updates_gpu, layer.weight_updates, layer.c*layer.n*layer.size*layer.size/layer.groups);
    cuda_pull_array(layer.bias_updates_gpu, layer.bias_updates, layer.n);
    if (layer.batch_normalize){
        cuda_pull_array(layer.scales_gpu, layer.scales, layer.n);
        cuda_pull_array(layer.rolling_mean_gpu, layer.rolling_mean, layer.n);
        cuda_pull_array(layer.rolling_variance_gpu, layer.rolling_variance, layer.n);
    }
}

void push_convolutional_layer(convolutional_layer layer)
{
    cuda_push_array(layer.weights_gpu, layer.weights, layer.c*layer.n*layer.size*layer.size/layer.groups);
    cuda_push_array(layer.biases_gpu, layer.biases, layer.n);
    cuda_push_array(layer.weight_updates_gpu, layer.weight_updates, layer.c*layer.n*layer.size*layer.size/layer.groups);
    cuda_push_array(layer.bias_updates_gpu, layer.bias_updates, layer.n);
    if (layer.batch_normalize){
        cuda_push_array(layer.scales_gpu, layer.scales, layer.n);
        cuda_push_array(layer.rolling_mean_gpu, layer.rolling_mean, layer.n);
        cuda_push_array(layer.rolling_variance_gpu, layer.rolling_variance, layer.n);
    }
}

void update_convolutional_layer_gpu(layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    int size = l.size*l.size*l.c*l.n / l.groups;

    if(a.adam){
        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, size, batch, a.t);
        adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
        if(l.scales_gpu){
            adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
        }
    }else{
        axpy_gpu(size, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
        axpy_gpu(size, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
        scal_gpu(size, momentum, l.weight_updates_gpu, 1);

        axpy_gpu(l.n, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
        scal_gpu(l.n, momentum, l.bias_updates_gpu, 1);

        if(l.scales_gpu){
            axpy_gpu(l.n, learning_rate/batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
            scal_gpu(l.n, momentum, l.scale_updates_gpu, 1);
        }
    }
}


