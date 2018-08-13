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
}

__global__ void binarize_kernel(float *x, int n, float *binary);


void binarize_gpu(float *x, int n, float *binary);

__global__ void binarize_input_kernel(float *input, int n, int size, float *binary);


void binarize_input_gpu(float *input, int n, int size, float *binary);



__global__ void binarize_weights_kernel(float *weights, int n, int size, float *binary);

void binarize_weights_gpu(float *weights, int n, int size, float *binary);


void forward_dilated_conv_layer_gpu(dilated_convolutional_layer l, network net)
{
    //fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
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
    //constrain_gpu(l.outputs*l.batch, 1, l.delta_gpu, 1);
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

            im2col_gpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
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



void test_dconv_backprop_gpu()
{
    
    int batch = 1;
    int h = 10;
    int w = 10;
    int c = 3;
    int n = 1;
    int groups = 1;
    int size = 3;
    int stride = 1;
    int padding = 3;
    ACTIVATION activation = LEAKY;
    int batch_normalize = 0;
    int binary = 0;
    int xnor = 0;
    int adam = 0;
    int dilate_rate = 2;
    
    // data: 10*10*3
    // weights: 3*3*3 -> 7*7*3, padding = 3
    // output: 10*10*3
    // delta: 10*10*3
    float data[] = {
        1,1,1,1,1,1,1,1,1,1,
        2,2,2,2,2,2,2,2,2,2,
        3,3,3,3,3,3,3,3,3,3,
        4,4,4,4,4,4,4,4,4,4,
        5,5,5,5,5,5,5,5,5,5,
        6,6,6,6,6,6,6,6,6,6,
        7,7,7,7,7,7,7,7,7,7,
        8,8,8,8,8,8,8,8,8,8,
        9,9,9,9,9,9,9,9,9,9,
        9,9,9,9,9,9,9,9,9,9,

        1,1,1,1,1,1,1,1,1,1,
        2,2,2,2,2,2,2,2,2,2,
        3,3,3,3,3,3,3,3,3,3,
        4,4,4,4,4,4,4,4,4,4,
        5,5,5,5,5,5,5,5,5,5,
        6,6,6,6,6,6,6,6,6,6,
        7,7,7,7,7,7,7,7,7,7,
        8,8,8,8,8,8,8,8,8,8,
        9,9,9,9,9,9,9,9,9,9,
        9,9,9,9,9,9,9,9,9,9,

        1,1,1,1,1,1,1,1,1,1,
        2,2,2,2,2,2,2,2,2,2,
        3,3,3,3,3,3,3,3,3,3,
        4,4,4,4,4,4,4,4,4,4,
        5,5,5,5,5,5,5,5,5,5,
        6,6,6,6,6,6,6,6,6,6,
        7,7,7,7,7,7,7,7,7,7,
        8,8,8,8,8,8,8,8,8,8,
        9,9,9,9,9,9,9,9,9,9,
        9,9,9,9,9,9,9,9,9,9};
    float weights[27] = {0};
        for(int i=0; i<27; i++) weights[i] = 1;
    float delta[10*10*1] = {0};
        for (int i=0; i<10*10*1; i++) delta[i] = i+1;
    float weight_updates[3*3*3];
        for (int i=0; i<3*3*3; i++) weight_updates[i] = 0;
    float upper_delta[10*10*3];
        for (int i=0; i<10*10*3; i++) upper_delta[i] = 0;
    float work[3*3*3*10*10*1] = {0};
        for (int i=0; i<3*3*3*10*10*1; i++) work[i] = 0;
    float output[10*10*1];
        for (int i=0; i<10*10*1; i++) output[i] = 0;
    
    
    dilated_convolutional_layer l = make_dilated_conv_layer(
        batch,h,w,c,n,groups,size,stride,padding,activation, batch_normalize, binary, xnor, adam, dilate_rate);
    
    network net = *make_network(1);
    net.layers = &l;

    cudaMalloc((void**)&l.output_gpu, 10*10*1*sizeof(float));
	cudaMalloc((void**)&l.weights_gpu, 3*3*3*sizeof(float));
	cudaMalloc((void**)&l.weight_updates_gpu, 3*3*3*sizeof(float));
	cudaMalloc((void**)&l.delta_gpu, 10*10*1*sizeof(float));
	cudaMalloc((void**)&net.input_gpu, 10*10*3*sizeof(float));
	cudaMalloc((void**)&net.workspace, 3*3*3*10*10*sizeof(float));
	cudaMalloc((void**)&net.delta_gpu, 10*10*3*sizeof(float));

    cudaMemcpy(l.output_gpu, output, 10*10*1*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(l.weights_gpu, weights, 3*3*3*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(l.weight_updates_gpu, weight_updates, 3*3*3*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.delta_gpu, delta, 10*10*1*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(net.input_gpu, data, 10*10*3*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(net.workspace, work, 3*3*3*10*10*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(net.delta_gpu, upper_delta, 10*10*3*sizeof(float),cudaMemcpyHostToDevice);
    forward_dilated_conv_layer_gpu(l, net);

    printf("forward dconv gpu complete.\n");

    cudaMemcpy(output, l.output_gpu, 10*10*sizeof(float),cudaMemcpyDeviceToHost);
    
    printf("GPU Output = 10*10*1\n");
    for(int i=0; i<10; i++){
        for(int j=0; j<10; j++){
            printf("%f\t",output[i*10+j]);
        }printf("\n");
    }printf("\n");

    backward_dilated_conv_layer_gpu(l,net);
    // l.weight_updates, net.delta
    cudaMemcpy(weight_updates, l.weight_updates_gpu, 3*3*3*sizeof(float),cudaMemcpyDeviceToHost);
    printf("GPU Weight Updates = \n");
    for(int i=0; i<3; i++){
        for(int j=0; j<3*3; j++){
            printf("%f\t", weight_updates[i*3+j]);
        }
        printf("\n");
    }printf("\n");

    printf("GPU Upper layer delta = \n");
    cudaMemcpy(upper_delta, net.delta_gpu, 10*10*3*sizeof(float),cudaMemcpyDeviceToHost);
    for(int i=0; i<10*3; i++){
        for(int j=0; j<10; j++){
            printf("%f\t", upper_delta[i*10+j]);
        }
        printf("\n");
    }printf("\n");
}
