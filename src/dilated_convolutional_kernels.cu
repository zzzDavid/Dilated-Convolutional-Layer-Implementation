#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "convolutional_layer.h"
#include "dilated_convolutional_layer.h"
#include "batchnorm_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col_dilated.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"
#include "darknet.h"
}

__global__ void binarize_kernel(float *x, int n, float *binary);


void binarize_gpu(float *x, int n, float *binary);

__global__ void binarize_input_kernel(float *input, int n, int size, float *binary);


void binarize_input_gpu(float *input, int n, int size, float *binary);



__global__ void binarize_weights_kernel(float *weights, int n, int size, float *binary);

void binarize_weights_gpu(float *weights, int n, int size, float *binary);


void forward_dilated_conv_layer_gpu(dilated_convolutional_layer l, network net)
{
    printf("I'm in forward_dilated_conv_layer_gpu!\n");
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    printf("Fill GPU success!\n");
    if(l.binary){
        printf("Binarize in progress!\n");
        binarize_weights_gpu(l.weights_gpu, l.n, l.c/l.groups*l.size*l.size, l.binary_weights_gpu);
        swap_binary(&l);
    }

    if(l.xnor){
        printf("Xnor construction in progress!\n");
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
                printf("I'm going to call im2col_dilated_gpu!\n");
                //-----------print im2col input-----------------------------------------------------
                /*printf("image = \n");
                float *temp = im;
                for (int i = 1; i <= l.inputs; i++)
                {
                    if (i % 10 == 0)
                    {
                        printf("%d\t", (int)*temp);
                        printf("\n");
                        temp = temp + 1;
                    }else{
                        printf("%d\t", (int)*temp);
                        temp = temp + 1;
                    }
                    //printf("i = %d\t", i);
                }*/
                //-----------------------------------------------------------------------------------
                im2col_dilated_gpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, l.dilate_rate, b);
                //------------print im2col output-----------------------------------------------------
                printf("image_col = \n");
                float *temp = b;
                for (int i = 1; i <= 36*12; i++)
                {
                    if (i % 36 == 0)
                    {
                        printf("%d  ", (int)*temp);
                        printf("\n");
                        temp = temp + 1;
                    }else{
                        printf("%d  ", (int)*temp);
                        temp = temp + 1;
                    }
                }
                //-------------------------------------------------------------------------------------

            }
            //printf("I'm going to call gemm_gpu!\n");
            gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
            //printf("gemm_gpu finished\n");
            // TA = 0; TB = 0, M = m, N = n, K = k, Alpha = 1, *A = a, lda = n, *B = b, ldb = k, Beta = 1, *C = c, ldc = n
            /*void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
                float *A, int lda, 
                float *B, int ldb,
                float BETA,
                float *C, int ldc)*/

            // print gemm output
            /*printf("gemm_gpu output = \n");
            float *temp = c;
            for (int i = 1; i <= l.outputs; i++)
            {
                if (i % 2 == 0)
                {
                    printf("%f\t", *temp);
                    printf("\n");
                    temp = temp + 1;
                }else{
                    printf("%f\t", *temp);
                    temp = temp + 1;
            }*/
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

            im2col_dilated_gpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, 3, b);
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

                if (l.size != 1) {
                    col2im_gpu(net.workspace, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, imd);
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

void test_dilated_conv_layer_gpu()
{
    printf("Entering test_dilated_conv_layer()\n");
    dilated_convolutional_layer l = make_dilated_conv_layer(1, 10, 10, 3, 1, 1, 2, 1, 0, LEAKY, 0, 0, 0, 0, 2);
    // batch = 1, h = 10, w = 10, c = 3, n = 1, group = 1, size = 2, stride = 1, padding = 0, activation = LEAKY, 
    // batch_nomarlize = 0, binary = 0, xnor = 0, adam = 0, dilate_rate = 2
    printf("make dilated conv layer success!\n");
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

    float w[] = {
        1,1,1,1,1,1,1,1,1,1,1,1
    };
    float out[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    network net = *make_network(1);
    net.layers = &l;
    net.input_gpu = data;
    net.workspace = (float*) calloc(1, l.outputs);
    l.weights_gpu = w;
    forward_dilated_conv_layer_gpu(l, net);
    
    //float *temp = out;
    //cudaMemcpy(temp, l.output_gpu, l.outputs*sizeof(float),cudaMemcpyDeviceToHost);
    //printf("Output:\n");
    //printf("Number of output: %d\n", l.outputs);
    /*for (int i = 1; i <= l.outputs; i++)
    {
        if (i % 6 == 0)
        {
            printf("%f\t", *temp);
            printf("\n");
            temp = temp + 1;
        }else{
            printf("%f\t", *temp);
            temp = temp + 1;
        }
        //printf("i = %d\t", i);
    }*/
}

