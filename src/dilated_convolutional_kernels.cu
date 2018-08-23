#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "convolutional_layer.h"
#include "dilated_convolutional_layer.h"
#include "batchnorm_layer.h"
#include "gemm.h"
#include "blas.h"
#include "utils.h"
#include "darknet.h"
#include "cuda.h"
#include <stdio.h>
#include <stdlib.h>
}

__global__ void binarize_kernel(float *x, int n, float *binary);


void binarize_gpu(float *x, int n, float *binary);

__global__ void binarize_input_kernel(float *input, int n, int size, float *binary);


void binarize_input_gpu(float *input, int n, int size, float *binary);



__global__ void binarize_weights_kernel(float *weights, int n, int size, float *binary);

void binarize_weights_gpu(float *weights, int n, int size, float *binary);


void forward_dilated_conv_layer_gpu(dilated_convolutional_layer l, network net)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    if(l.binary){
        binarize_weights_gpu(l.weights_gpu, l.n, l.c/l.groups*l.size*l.size, l.binary_weights_gpu);
        swap_binary(&l);
    }

    if(l.xnor){
        binarize_weights_gpu(l.weights_gpu, l.n, l.c/l.groups*l.size*l.size, l.binary_weights_gpu);
        swap_binary(&l);
        binarize_gpu(net.input_gpu, l.c*l.h*l.w*l.batch, l.binary_input_gpu);
        net.input_gpu = l.binary_input_gpu;
    }

#ifdef CUDNN
    float one = 1;
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
                &one,
                l.dstTensorDesc,
                l.output_gpu);

#else
    int i, j;
    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_w*l.out_h;
    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.weights_gpu + j*l.nweights/l.groups;
            float *b = net.workspace;
            float *c = l.output_gpu + (i*l.groups + j)*n*m;
            float *im = net.input_gpu + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            if (l.size == 1){
                b = im;
            } else {
                im2col_dilated_gpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, l.dilate_rate, b);
            }
            gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }

    }
    
#endif

    if (l.batch_normalize) {
        forward_batchnorm_layer_gpu(l, net);
    } else {
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
    }

    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
    //if(l.dot > 0) dot_error_gpu(l);
    if(l.binary || l.xnor) swap_binary(&l);
}

__global__ void smooth_kernel(float *x, int n, int w, int h, int c, int size, float rate, float *delta);


extern "C" void smooth_layer(layer l, int size, float rate);


void backward_dilated_conv_layer_gpu(convolutional_layer l, network net)
{
    if(l.smooth){
        smooth_layer(l, 5, l.smooth);
    } 
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
    int m = l.n/l.groups;
    int n = l.size*l.size*l.c/l.groups;
    int k = l.out_w*l.out_h;

    int i, j;
    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.delta_gpu + (i*l.groups + j)*m*k;
            float *b = net.workspace;
            float *c = l.weight_updates_gpu + j*l.nweights/l.groups;

            float *im  = net.input_gpu+(i*l.groups + j)*l.c/l.groups*l.h*l.w;
            float *imd = net.delta_gpu+(i*l.groups + j)*l.c/l.groups*l.h*l.w;

            im2col_dilated_gpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad,l.dilate_rate, b);
            gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);

            if (net.delta_gpu) {
                if (l.binary || l.xnor) swap_binary(&l);
                a = l.weights_gpu + j*l.nweights/l.groups;
                b = l.delta_gpu + (i*l.groups + j)*m*k;
                c = net.workspace;
                if (l.size == 1) {
                    c = imd;
                }

                gemm_gpu(1,0,n,k,m,1,a,n,b,k,0,c,k);
                //------------------------------------------------------------
                /*printf("GPU input of col2im_dilated = \n");
                float input[n*k];
                cudaMemcpy(input, c, n*k*sizeof(float),cudaMemcpyDeviceToHost);
                for (int i=0; i<n; i++){
                    for (int j=0; j<k; j++){
                        printf("%d ",(int)input[i*k+j]);
                    }printf("\n");
                }printf("\n");*/
                //------------------------------------------------------------
                if (l.size != 1) {
                    col2im_dilated_gpu(net.workspace, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, l.dilate_rate, imd);
                //-----------------------------------------------------------    
                    /*printf("GPU output of col2im_dilated = \n");
                    float output[l.h*l.c*l.w];
                    cudaMemcpy(output, imd, l.h*l.w*l.c, cudaMemcpyDeviceToHost);
                    for (int i=0; i<l.h*l.c; i++){
                        for (int j=0; j<l.w; j++){
                            printf("%f\t",output[i*l.w+j]);
                        }printf("\n");
                    }printf("\n");*/
                //------------------------------------------------------------
                }
                if(l.binary || l.xnor) {
                    swap_binary(&l);
                }
            }
            if(l.xnor) gradient_array_gpu(original_input + i*l.c*l.h*l.w, l.c*l.h*l.w, HARDTAN, net.delta_gpu + i*l.c*l.h*l.w);
        }
    }
#endif
}

void pull_dilated_conv_layer(layer l)
{
    cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
    cuda_pull_array(l.biases_gpu, l.biases, l.n);
    cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
    cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
    if (l.batch_normalize){
        cuda_pull_array(l.scales_gpu, l.scales, l.n);
        cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}

void push_dilated_conv_layer(layer l)
{
    cuda_push_array(l.weights_gpu, l.weights, l.nweights);
    cuda_push_array(l.biases_gpu, l.biases, l.n);
    cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
    cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
    if (l.batch_normalize){
        cuda_push_array(l.scales_gpu, l.scales, l.n);
        cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}


void update_dilated_conv_layer_gpu(layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    if(a.adam){
        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.nweights, batch, a.t);
        adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
        if(l.scales_gpu){
            adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
        }
    }else{
        axpy_gpu(l.nweights, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
        axpy_gpu(l.nweights, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
        scal_gpu(l.nweights, momentum, l.weight_updates_gpu, 1);

        axpy_gpu(l.n, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
        scal_gpu(l.n, momentum, l.bias_updates_gpu, 1);

        if(l.scales_gpu){
            axpy_gpu(l.n, learning_rate/batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
            scal_gpu(l.n, momentum, l.scale_updates_gpu, 1);
        }
    }
    if(l.clip){
        constrain_gpu(l.nweights, l.clip, l.weights_gpu, 1);
    }
}

void test_dconv_forward_gpu()
{
    
    int batch = 100;
    int h = 32;
    int w = 32;
    int c = 3;
    int n = 32;
    int groups = 1;
    int size = 5;
    int stride = 1;
    int padding = 5;
    ACTIVATION activation = LEAKY;
    int batch_normalize = 0;
    int binary = 0;
    int xnor = 0;
    int adam = 0;
    int dilate_rate = 2;
    
    
    dilated_convolutional_layer l = make_dilated_conv_layer(
        batch,h,w,c,n,groups,size,stride,padding,activation, batch_normalize, binary, xnor, adam, dilate_rate);
    
    network net = *make_network(1);
    net.layers = &l;

    float *input_cpu, *weights_cpu, *output_cpu;
	input_cpu = (float*) calloc (batch*h*w*c, sizeof(float));
	weights_cpu = (float*) calloc (size*size*c*n, sizeof(float));
    output_cpu = (float*) calloc (batch*l.out_c*l.out_h*l.out_w, sizeof(float));
    
    FILE *fp;
	    if((fp=fopen("caffe_forward_input.txt","r"))==NULL){
			printf("Open file caffe_forward_input failed.\n");
			exit(0);
		}

		for(int i=0; i<h*w*c*batch; i++){
			fscanf(fp,"%f,", &input_cpu[i]);
		}
		fclose(fp);


		FILE *fin;
		if ((fin = fopen("caffe_forward_weights.txt","r"))==NULL){
			printf("Open file caffe_forward_weights failed.\n");
			exit(0);
		}
		//fscanf(fin, "%*[^\n]\n", NULL,NULL);
		for(int i=0; i<size*size*c*n; i++){
			fscanf(fin, "%f,", &weights_cpu[i]);
		}
		fclose(fin);
    printf("finish reading all inputs.\n");

    cudaMalloc((void**)&l.output_gpu, batch*l.out_w*l.out_h*l.out_c*sizeof(float));
	cudaMalloc((void**)&l.weights_gpu, size*size*c*n*sizeof(float));
	cudaMalloc((void**)&net.input_gpu, batch*h*w*c*sizeof(float));
	cudaMalloc((void**)&net.workspace, batch*size*size*c*l.out_w*l.out_h*sizeof(float));


    cudaMemcpy(l.output_gpu, output_cpu, batch*l.out_w*l.out_h*l.out_c*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(l.weights_gpu, weights_cpu, size*size*c*n*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(net.input_gpu, input_cpu, batch*h*w*c*sizeof(float),cudaMemcpyHostToDevice);
    
    forward_dilated_conv_layer_gpu(l, net);

    printf("forward dconv gpu complete.\n");

    cudaMemcpy(output_cpu, l.output_gpu, batch*l.out_c*l.out_w*l.out_h*sizeof(float),cudaMemcpyDeviceToHost);
    
    FILE *f3;
	if((f3 = fopen("darknet_output.txt", "a"))==NULL){
		printf("Error opening file darknet_output\n");
		exit(0);
	}
	for (int i=0; i<l.out_c*l.out_h*l.out_w*batch; i++){
		fprintf(f3, "%e, ", output_cpu[i]);
		if (i%10 == 9) fprintf(f3,"\n");
	}
    fclose(f3);
    
    printf("test completed successfully.\n");
}


void test_dconv_backprop_gpu()
{
    
    int batch = 100;
    int h = 8;
    int w = 8;
    int c = 32;
    int n = 64;
    int groups = 1;
    int size = 5;
    int stride = 1;
    int padding = 5;
    ACTIVATION activation = LEAKY;
    int batch_normalize = 0;
    int binary = 0;
    int xnor = 0;
    int adam = 0;
    int dilate_rate = 2;
    
    
    dilated_convolutional_layer l = make_dilated_conv_layer(
        batch,h,w,c,n,groups,size,stride,padding,activation, batch_normalize, binary, xnor, adam, dilate_rate);
    
    network net = *make_network(1);
    net.layers = &l;

    float *input_cpu, *weights_cpu, *delta_cpu, *weight_updates_cpu, *upperdelta_cpu, *output_cpu;
	input_cpu = (float*) calloc (batch*h*w*c, sizeof(float));
	weights_cpu = (float*) calloc (size*size*c*n, sizeof(float));
	delta_cpu = (float*) calloc (batch*l.out_w*l.out_h*l.out_c, sizeof(float));
	upperdelta_cpu = (float*) calloc (batch*h*w*c, sizeof(float));
    weight_updates_cpu = (float*) calloc (size*size*c*n, sizeof(float));
    output_cpu = (float*) calloc (batch*l.out_c*l.out_h*l.out_w, sizeof(float));
    
    FILE *fp;
	    if((fp=fopen("caffe_backprop_input.txt","r"))==NULL){
			printf("Open file caffe_backprop_input failed.\n");
			exit(0);
		}

		for(int i=0; i<h*w*c*batch; i++){
			fscanf(fp,"%f,", &input_cpu[i]);
		}
		fclose(fp);


		FILE *fin;
		if ((fin = fopen("caffe_backprop_weights.txt","r"))==NULL){
			printf("Open file caffe_backprop_weights failed.\n");
			exit(0);
		}
		//fscanf(fin, "%*[^\n]\n", NULL,NULL);
		for(int i=0; i<size*size*c*n; i++){
			fscanf(fin, "%f,", &weights_cpu[i]);
		}
		fclose(fin);

		FILE *f1;
		if ((f1 = fopen("caffe_backprop_topdiff.txt","r"))==NULL){
			printf("Open file caffe_backprop_topdiff.txt failed.\n");
			exit(0);
		}
		for (int i=0; i<l.out_w*l.out_h*l.out_c*batch; i++){
			fscanf(f1, "%f,", &delta_cpu[i]);
		}
        fclose(f1);
    printf("finish reading all inputs.\n");

    cudaMalloc((void**)&l.output_gpu, batch*l.out_w*l.out_h*l.out_c*sizeof(float));
	cudaMalloc((void**)&l.weights_gpu, size*size*c*n*sizeof(float));
	cudaMalloc((void**)&l.weight_updates_gpu, size*size*c*n*sizeof(float));
	cudaMalloc((void**)&l.delta_gpu, batch*l.out_w*l.out_h*l.out_c*sizeof(float));
	cudaMalloc((void**)&net.input_gpu, batch*h*w*c*sizeof(float));
	cudaMalloc((void**)&net.workspace, batch*size*size*c*l.out_w*l.out_h*sizeof(float));
	cudaMalloc((void**)&net.delta_gpu, batch*c*h*w*sizeof(float));

    cudaMemcpy(l.output_gpu, output_cpu, batch*l.out_w*l.out_h*l.out_c*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(l.weights_gpu, weights_cpu, size*size*c*n*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(l.weight_updates_gpu, weight_updates_cpu, size*size*c*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.delta_gpu, delta_cpu, batch*l.out_w*l.out_h*l.out_c*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(net.input_gpu, input_cpu, batch*h*w*c*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(net.delta_gpu, upperdelta_cpu, c*h*w*sizeof(float),cudaMemcpyHostToDevice);

    //forward_dilated_conv_layer_gpu(l, net);

    //printf("forward dconv gpu complete.\n");

    cudaMemcpy(output_cpu, l.output_gpu, batch*l.out_c*l.out_w*l.out_h*sizeof(float),cudaMemcpyDeviceToHost);
    


    backward_dilated_conv_layer_gpu(l,net);
    printf("backprop dconv gpu complete.\n");

    cudaMemcpy(weight_updates_cpu, l.weight_updates_gpu, size*size*c*n*sizeof(float),cudaMemcpyDeviceToHost);

    cudaMemcpy(upperdelta_cpu, net.delta_gpu, batch*h*w*c*sizeof(float),cudaMemcpyDeviceToHost);

    FILE *f;
	if((f = fopen("darknet_weight_diff.txt", "a"))==NULL){
		printf("Error opening file weight_diff\n");
		exit(0);
	}
	for (int i=0; i<size*size*n*c; i++){
		fprintf(f,"%e,",weight_updates_cpu[i]);
		if (i%10 == 9) fprintf(f,"\n");
	}
	fclose(f);

	FILE *f2;
	if((f2 = fopen("darknet_bottom_diff.txt", "a"))==NULL){
		printf("Error opening file bottom_diff\n");
		exit(0);
	}
	for (int i=0; i<h*w*c*batch; i++){
		fprintf(f2, "%e, ", upperdelta_cpu[i]);
		if (i%10 == 9) fprintf(f2,"\n");
	}
    fclose(f2);
    
    printf("test completed successfully.\n");
}
