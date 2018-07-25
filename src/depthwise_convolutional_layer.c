#include "depthwise_convolutional_layer.h"
#include "utils.h"
#include "batchnorm_layer.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>

#ifdef AI2
#include "xnor_layer.h"
#endif

/*
void swap_binary(depthwise_convolutional_layer *l)
{
    float *swap = l->weights;
    l->weights = l->binary_weights;
    l->binary_weights = swap;

#ifdef GPU
    swap = l->weights_gpu;
    l->weights_gpu = l->binary_weights_gpu;
    l->binary_weights_gpu = swap;
#endif
}

void binarize_weights(float *weights,int c, int size, float *binary)
{
    int i, f;
    for(f = 0; f < c; ++f){
        float mean = 0;
        for(i = 0; i < size; ++i){
            mean += fabs(weights[f*size + i]);
        }
        mean = mean / size;
        for(i = 0; i < size; ++i){
            binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        }
    }
}

void binarize_cpu(float *input, int n, float *binary)
{
    int i;
    for(i = 0; i < n; ++i){
        binary[i] = (input[i] > 0) ? 1 : -1;
    }
}

void binarize_input(float *input, int n, int size, float *binary)
{
    int i, s;
    for(s = 0; s < size; ++s){
        float mean = 0;
        for(i = 0; i < n; ++i){
            mean += fabs(input[i*size + s]);
        }
        mean = mean / n;
        for(i = 0; i < n; ++i){
            binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
        }
    }
}
*/

int depthwise_convolutional_out_height(depthwise_convolutional_layer l)
{
    return (l.h + 2*l.pad - l.size) / l.stride + 1;
}

int depthwise_convolutional_out_width(depthwise_convolutional_layer l)
{
    return (l.w + 2*l.pad - l.size) / l.stride + 1;
}

image get_depthwise_convolutional_image(depthwise_convolutional_layer l)
{
    return float_to_image(l.out_w,l.out_h,l.out_c,l.output);
}

image get_depthwise_convolutional_delta(depthwise_convolutional_layer l)
{
    return float_to_image(l.out_w,l.out_h,l.out_c,l.delta);
}

static size_t get_workspace_size(layer l){
#ifdef CUDNN
    if(gpu_index >= 0){
        size_t most = 0;
        size_t s = 0;
        cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.weightDesc,
                l.convDesc,
                l.dstTensorDesc,
                l.fw_algo,
                &s);
        if (s > most) most = s;
        cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dweightDesc,
                l.bf_algo,
                &s);
        if (s > most) most = s;
        cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle(),
                l.weightDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dsrcTensorDesc,
                l.bd_algo,
                &s);
        if (s > most) most = s;
        return most;
    }
#endif
    return (size_t)l.out_h*l.out_w*l.size*l.size*l.c*sizeof(float);
}

/////////////////////////////////TODO
#ifdef GPU
#ifdef CUDNN
void cudnn_depthwise_convolutional_setup(layer *l)
{
    cudnnSetTensor4dDescriptor(l->dsrcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w); 
    cudnnSetTensor4dDescriptor(l->ddstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 
    cudnnSetFilter4dDescriptor(l->dweightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l->n, l->c, l->size, l->size); 

    cudnnSetTensor4dDescriptor(l->srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w); 
    cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 
    cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1); 
    cudnnSetFilter4dDescriptor(l->weightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l->n, l->c, l->size, l->size); 
    // cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION);
    #if CUDNN_MAJOR >= 6
    cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    #else
    cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION);
    #endif
    cudnnGetConvolutionForwardAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->weightDesc,
            l->convDesc,
            l->dstTensorDesc,
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
            0,
            &l->fw_algo);
    cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle(),
            l->weightDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dsrcTensorDesc,
            CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
            0,
            &l->bd_algo);
    cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dweightDesc,
            CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
            0,
            &l->bf_algo);
}
#endif
#endif
///////////////////////////////////////////

depthwise_convolutional_layer make_depthwise_convolutional_layer(int batch, int h, int w, int c, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam)
{
    int i;
    depthwise_convolutional_layer l = {0};
    l.type = DEPTHWISE_CONVOLUTIONAL;

    l.h = h;
    l.w = w;
    l.c = c;
    //l.n = n;
    //l.binary = binary;
    //l.xnor = xnor;
    l.batch = batch;
    l.stride = stride;
    l.size = size;
    l.pad = padding;
    l.batch_normalize = batch_normalize;

    l.weights = calloc(c*size*size, sizeof(float));///////////////////////
    l.weight_updates = calloc(c*size*size, sizeof(float));//////////////////////

    //l.biases = calloc(n, sizeof(float));
    //l.bias_updates = calloc(n, sizeof(float));

    l.nweights = c*size*size;//////////////
    //l.nbiases = n;

    // float scale = 1./sqrt(size*size*c);
    float scale = sqrt(2./(size*size*c));
    //scale = .02;
    //for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_uniform(-1, 1);
    for(i = 0; i < c*size*size; ++i) l.weights[i] = scale*rand_normal();////////////////////
    int out_w = depthwise_convolutional_out_width(l);
    int out_h = depthwise_convolutional_out_height(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = c;///////////////////
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.output = calloc(l.batch*l.outputs, sizeof(float));
    l.delta  = calloc(l.batch*l.outputs, sizeof(float));

    l.forward = forward_depthwise_convolutional_layer;
    l.backward = backward_depthwise_convolutional_layer;
    l.update = update_depthwise_convolutional_layer;
    /*
    if(binary){
        l.binary_weights = calloc(c*size*size, sizeof(float));////////////////////////
        l.cweights = calloc(c*size*size, sizeof(char));/////////////////////////
        l.scales = calloc(c, sizeof(float));////////////////////////
    }
    if(xnor){
        l.binary_weights = calloc(c*size*size, sizeof(float));////////////////////////
        l.binary_input = calloc(l.inputs*l.batch, sizeof(float));
    }
     */

    if(batch_normalize){
        l.scales = calloc(c, sizeof(float));//////////////////
        l.scale_updates = calloc(c, sizeof(float));///////////////////////
        for(i = 0; i < c; ++i){//////////////////////
            l.scales[i] = 1;
        }

        l.mean = calloc(c, sizeof(float));////////////////////
        l.variance = calloc(c, sizeof(float));//////////////

        l.mean_delta = calloc(c, sizeof(float));///////////////
        l.variance_delta = calloc(c, sizeof(float));////////////////

        l.rolling_mean = calloc(c, sizeof(float));/////////////
        l.rolling_variance = calloc(c, sizeof(float));/////////////////
        l.x = calloc(l.batch*l.outputs, sizeof(float));
        l.x_norm = calloc(l.batch*l.outputs, sizeof(float));
    }
    if(adam){
        //l.m = calloc(c*size*size, sizeof(float));/////////////////////
        //l.v = calloc(c*size*size, sizeof(float));///////////////
        //l.bias_m = calloc(c, sizeof(float));///////////////
        //l.scale_m = calloc(c, sizeof(float));//////////////
        //l.bias_v = calloc(c, sizeof(float));////////////////
        //l.scale_v = calloc(c, sizeof(float));/////////////////
    }

////////////////////////////////////TODO
#ifdef GPU
    l.forward_gpu = forward_depthwise_convolutional_layer_gpu;
    l.backward_gpu = backward_depthwise_convolutional_layer_gpu;
    l.update_gpu = update_depthwise_convolutional_layer_gpu;

    if(gpu_index >= 0){
        if (adam) {
            //l.m_gpu = cuda_make_array(l.m, c*size*size);
            //l.v_gpu = cuda_make_array(l.v, c*size*size);
            //l.bias_m_gpu = cuda_make_array(l.bias_m, c);
            //l.bias_v_gpu = cuda_make_array(l.bias_v, c);
            //l.scale_m_gpu = cuda_make_array(l.scale_m, c);
            //l.scale_v_gpu = cuda_make_array(l.scale_v, c);
        }

        l.weights_gpu = cuda_make_array(l.weights, c*size*size);///////////////
        l.weight_updates_gpu = cuda_make_array(l.weight_updates, c*size*size);////////////////

        //l.biases_gpu = cuda_make_array(l.biases, c);
        //l.bias_updates_gpu = cuda_make_array(l.bias_updates, c);

        l.delta_gpu = cuda_make_array(l.delta, l.batch*out_h*out_w*c);//////////////////////
        l.output_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*c);//////////////////////

/*
        if(binary){
            l.binary_weights_gpu = cuda_make_array(l.weights, c*size*size);
        }
        if(xnor){
            l.binary_weights_gpu = cuda_make_array(l.weights, c*size*size);
            l.binary_input_gpu = cuda_make_array(0, l.inputs*l.batch);
        }
        */

        if(batch_normalize){
            l.mean_gpu = cuda_make_array(l.mean, c);////////////////
            l.variance_gpu = cuda_make_array(l.variance, c);/////////////

            l.rolling_mean_gpu = cuda_make_array(l.mean, c);//////////////////
            l.rolling_variance_gpu = cuda_make_array(l.variance, c);/////////////////

            l.mean_delta_gpu = cuda_make_array(l.mean, c);////////////////////
            l.variance_delta_gpu = cuda_make_array(l.variance, c);///////////////////

            l.scales_gpu = cuda_make_array(l.scales, c);////////////////////
            l.scale_updates_gpu = cuda_make_array(l.scale_updates, c);//////////////////////

            l.x_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*c);//////////////
            l.x_norm_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*c);///////////////////////
        }
#ifdef CUDNN
        cudnnCreateTensorDescriptor(&l.normTensorDesc);
        cudnnCreateTensorDescriptor(&l.srcTensorDesc);
        cudnnCreateTensorDescriptor(&l.dstTensorDesc);
        cudnnCreateFilterDescriptor(&l.weightDesc);
        cudnnCreateTensorDescriptor(&l.dsrcTensorDesc);
        cudnnCreateTensorDescriptor(&l.ddstTensorDesc);
        cudnnCreateFilterDescriptor(&l.dweightDesc);
        cudnnCreateConvolutionDescriptor(&l.convDesc);
        cudnn_depthwise_convolutional_setup(&l);
#endif
    }
#endif
////////////////////////////////////
    l.workspace_size = get_workspace_size(l);
    l.activation = activation;

    fprintf(stderr, "depth_conv  %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);

    return l;
}

void denormalize_depthwise_convolutional_layer(depthwise_convolutional_layer l)
{
    int i, j;
    for(i = 0; i < l.c; ++i){///////////////////////
        float scale = l.scales[i]/sqrt(l.rolling_variance[i] + .00001);
        for(j = 0; j < l.size*l.size; ++j){///////////////////////
            l.weights[i*l.size*l.size + j] *= scale;//////////////////////////
        }
        //l.biases[i] -= l.rolling_mean[i] * scale;
        l.scales[i] = 1;
        l.rolling_mean[i] = 0;
        l.rolling_variance[i] = 1;
    }
}

/*
void test_depthwise_convolutional_layer()
{
    depthwise_convolutional_layer l = make_depthwise_convolutional_layer(1, 5, 5, 3, 5, 2, 1, LEAKY, 1, 0, 0, 0);
    l.batch_normalize = 1;
    float data[] = {1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3};
    //net.input = data;
    //forward_depthwise_convolutional_layer(l);
}
*/

void resize_depthwise_convolutional_layer(depthwise_convolutional_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    int out_w = depthwise_convolutional_out_width(*l);
    int out_h = depthwise_convolutional_out_height(*l);

    l->out_w = out_w;
    l->out_h = out_h;

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->w * l->h * l->c;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta  = realloc(l->delta,  l->batch*l->outputs*sizeof(float));
    if(l->batch_normalize){
        l->x = realloc(l->x, l->batch*l->outputs*sizeof(float));
        l->x_norm  = realloc(l->x_norm, l->batch*l->outputs*sizeof(float));
    }

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =  cuda_make_array(l->delta,  l->batch*l->outputs);
    l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);

    if(l->batch_normalize){
        cuda_free(l->x_gpu);
        cuda_free(l->x_norm_gpu);

        l->x_gpu = cuda_make_array(l->output, l->batch*l->outputs);
        l->x_norm_gpu = cuda_make_array(l->output, l->batch*l->outputs);
    }
#ifdef CUDNN
    cudnn_depthwise_convolutional_setup(l);
#endif
#endif
    l->workspace_size = get_workspace_size(*l);
}

/*
void add_bias(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

void scale_bias(float *output, float *scales, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}

void backward_bias(float *bias_updates, float *delta, int batch, int n, int size)
{
    int i,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            bias_updates[i] += sum_array(delta+size*(i+b*n), size);
        }
    }
}
*/

void forward_depthwise_convolutional_layer(depthwise_convolutional_layer l, network net)
{
    int out_h = l.out_h;
    int out_w = l.out_w;
    int i;

    fill_cpu(l.outputs*l.batch, 0, l.output, 1);

    /*
    if(l.xnor){
        binarize_weights(l.weights, l.c, l.c*l.size*l.size, l.binary_weights);
        swap_binary(&l);
        binarize_cpu(net.input, l.c*l.h*l.w*l.batch, l.binary_input);
        net.input = l.binary_input;
    }
     */

    int m = l.c;
    int k = l.size*l.size*l.c;//////////////???????? l.c
    int n = out_h*out_w;


    float *a = l.weights;
    float *b = net.workspace;
    float *c = l.output;

    for(i = 0; i < l.batch; ++i){
        im2col_cpu(net.input, l.c, l.h, l.w, 
                l.size, l.stride, l.pad, b);
        gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);///////////////////////////////////????
        c += n*m;
        net.input += l.c*l.h*l.w;
    }

    if(l.batch_normalize){
        forward_batchnorm_layer(l, net);
    } else {
        //add_bias(l.output, l.biases, l.batch, l.n, out_h*out_w);
    }

    activate_array(l.output, m*n*l.batch, l.activation);
    //if(l.binary || l.xnor) swap_binary(&l);
}

void backward_depthwise_convolutional_layer(depthwise_convolutional_layer l, network net)
{
    int i;
    int m = l.c;
    int n = l.size*l.size*l.c;////////////////////////////////??? l.c
    int k = l.out_w*l.out_h;

    gradient_array(l.output, m*k*l.batch, l.activation, l.delta);

    if(l.batch_normalize){
        backward_batchnorm_layer(l, net);
    } else {
        //backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);
    }

    for(i = 0; i < l.batch; ++i){
        float *a = l.delta + i*m*k;
        float *b = net.workspace;
        float *c = l.weight_updates;

        float *im = net.input+i*l.c*l.h*l.w;

        im2col_cpu(im, l.c, l.h, l.w, 
                l.size, l.stride, l.pad, b);
        gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);////////////////////convolutional

        if(net.delta){
            a = l.weights;
            b = l.delta + i*m*k;
            c = net.workspace;

            gemm(1,0,n,k,m,1,a,n,b,k,0,c,k);////////////////////////

            col2im_cpu(net.workspace, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, net.delta+i*l.c*l.h*l.w);
        }
    }
}

void update_depthwise_convolutional_layer(depthwise_convolutional_layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    int size = l.size*l.size*l.c;//////////////////
    axpy_cpu(l.c, learning_rate/batch, l.bias_updates, 1, l.biases, 1);//////////////////??? l.n bias
    scal_cpu(l.c, momentum, l.bias_updates, 1);//////////////////??? l.n bias

    if(l.scales){
        axpy_cpu(l.c, learning_rate/batch, l.scale_updates, 1, l.scales, 1);//////////////////??? l.n bias
        scal_cpu(l.c, momentum, l.scale_updates, 1);//////////////////??? l.n bias
    }

    axpy_cpu(size, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(size, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(size, momentum, l.weight_updates, 1);
}


image get_depthwise_convolutional_weight(depthwise_convolutional_layer l, int i)
{
    int h = l.size;
    int w = l.size;
    int c = l.c;
    return float_to_image(w,h,c,l.weights+i*h*w*c);
}

void rgbgr_depthwise_weights(depthwise_convolutional_layer l)
{
    int i;
    for(i = 0; i < l.c; ++i){
        image im = get_depthwise_convolutional_weight(l, i);
        if (im.c == 3) {
            rgbgr_image(im);
        }
    }
}

void rescale_depthwise_weights(depthwise_convolutional_layer l, float scale, float trans)
{
    int i;
    for(i = 0; i < l.c; ++i){
        image im = get_depthwise_convolutional_weight(l, i);
        if (im.c == 3) {
            scale_image(im, scale);
            float sum = sum_array(im.data, im.w*im.h*im.c);
            //l.biases[i] += sum*trans;
        }
    }
}

image *get_depthwise_weights(depthwise_convolutional_layer l)
{
    image *weights = calloc(l.c, sizeof(image));///////////////////
    int i;
    for(i = 0; i < l.c; ++i){/////////////////////
        weights[i] = copy_image(get_depthwise_convolutional_weight(l, i));
        normalize_image(weights[i]);
        /*
        char buff[256];
        sprintf(buff, "filter%d", i);
        save_image(weights[i], buff);
        */
    }
    //error("hey");
    return weights;
}

image *visualize_depthwise_convolutional_layer(depthwise_convolutional_layer l, char *window, image *prev_weights)
{
    image *single_weights = get_depthwise_weights(l);
    show_images(single_weights, l.n, window);

    image delta = get_depthwise_convolutional_image(l);
    image dc = collapse_image_layers(delta, 1);
    char buff[256];
    sprintf(buff, "%s: Output", window);
    //show_image(dc, buff);
    //save_image(dc, buff);
    free_image(dc);
    return single_weights;
}

