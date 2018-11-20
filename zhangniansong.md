
# Dilated Convolutiaonal Layer Implemented on Darknet

## Modified

* `darknet.h`
  * In `typedef enum{}layer_type`: added `DILATED_CONVOLUTIONAL`.
  * In definition of `struct layer{}` added parameter `int dilate_rate`

* `darknet.c`
  * In functions `average()`, `numops()`, `rescale_net()`, `rbgr_net()`, `reset_normalize_net()`, `normalize_net()`, `denormalize_net()` added dilated conv support.
  * Added call to test function in `main`:
    ```C
    test_dconv_forward_cpu();
    test_dconv_backprop_cpu();
    ```

* `parser.c`
  * Add header: `dialted_convolutional_layer.h`.
  * In function: `string_to_layer_type()` added dilated conv support.
  * Add fucntion: `parse_dialted_convolutional()`.
  * In function: `parse_network_cfg()` added dilated conv support.
  * In function `save_weights_upto()`added support for dilated conv:
    ```C
    if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL ||l.type == DILATED_CONVOLUTIONAL){
        save_convolutional_weights(l.fp);
    }
    ```
  * In function `load_weights_upto()` added support for dilated conv:
    ```C
    if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL || l.type == DILATED_CONVOLUTIONAL){
            load_convolutional_weights(l, fp);
        }
    ```
* `network.c`
  * Add header: `dilated_convolutional_layer.h`.
  * In function `*get_layer_string()` added dilated conv support.
  * In function `resieze_network()` added dilated conv support.
  * In function `visualize_network` added dilated conv support.
  * In function `merge_weights`, `scale_weights`, `pull_weights`, `push_weights`, `distribute_weights` added dilated conv support.

* `gemm.c`
  * modified function `gemm_gpu`:
    ```C
    void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A_gpu, int lda,
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc)
    {
        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N), (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);
        cublasDestroy(handle);
    }
    ```
* `Makefile`
  * added C objects
    ```C
    OBJ=dilated_convolutional_layer.o im2col_dilated.o col2im_dilated.o`
    ```
  * added cuda objects:
    ```C
    OBJ+=dilated_convolutional_kernels.o im2col_kernels_dilated.o col2im_kernels_dilated.o
    ```

## Added files

* `im2col` function:
  * `im2col_dialted.h`
  * `im2col_dilated.c` : with dependence of function `im2col_get_pixel()` defined in `im2col.c`.
  * `im2col_dilated.cu`

* `col2im` function:
  * `col2im_dilated.h`
  * `col2im_dilated.c`
  * `col2im_dilated.cu`

* dilated convolutional layer implementation:
  * `dilated_convolutional_layer.h`
  * `dilated_convolutional_layer.c`
  * `dilated_convolutional_kernels.cu`
* `yolo-d-tiny.cfg`: yolov3-tiny with dilated convolutional layer.

## Test menthod

* Dump input and output from caffe
  * Added dumping functions in `conv)layer.cu`: `Forward_gpu()`, `Backward_gpu`;
  * Modified `cifar10_quick_train_test.prototxt`to add dilation.
  * To dump files, run:
    ```bash
    cd caffe
    make all
    ./examples/cifar10/train_quick.sh
    ```
  * dumped files: `caffe_forward_input.txt`, `caffe_forward_weights.txt`, `caffe_forward_output.txt`, `caffe_backprop_input.txt`, `caffe_backprop_weights.txt`, `caffe_backprop_top_diff.txt`, `caffe_backprop_bottom_diff.txt`, `caffe_backprop_weight_diff.txt`.

* Copy `caffe_forward_input.txt`, `caffe_forward_weights.txt`,  `caffe_backprop_input.txt`, `caffe_backprop_weights.txt`, `caffe_backprop_top_diff.txt` to the path of `darknet`.
* Run `./darknet`.

## Address

* Dilated convolutional layer: `niansong@192.168.1.7/home/niansong/dilated_convolution/darknet-work`

* Caffe used for dumping files: `niansong@192.168.1.7/home/niansong/dialted_convolution/caffe`

```c
int im_row = h_offset * dilate_rate + h * stride;
int im_col = w_offset * dilate_rate + w * stride;
int col_index = c * height_col * width_col + h * width_col + w;
```
