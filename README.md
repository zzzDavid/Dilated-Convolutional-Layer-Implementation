# An Implementation of Dilated Convolutional Layer Based on Darknet Architecture
This implementation of dilated convolutional layer was accomplished during my internship in Zerotech Inc., Beijing. The need for dilated convolutional layer comes from training `CSRNet` for crowd detection.

## About Darknet
Darknet is an open source neural network framework written in C and CUDA by Joseph Redmon. It is fast, easy to install, and supports CPU and GPU computation. For more information see the [Darknet project website](http://pjreddie.com/darknet).
### Darknet Dependency
Darknet has only two dependencies, and they are optional:
```
* OpenCV - for video I/O and more image types support
* CUDA - for GPU acceleration
```
This dilated convolutional layer implementation on GPU was written with `CUDA Toolkit 9.0`

### Darknet Installation
The `Makefile` is provided in the main folder, to install Darknet with dilated convolutional layer, type:
```
make
```

## Dilated Convolutional Layer

The added layer was tested by comparing with the output of `Caffe`'s implementation of dilated conv layer, and it was also tested by the training of `CSRNet` composed of it.

### Configuring the Dilated Convolutional Layer
Darknet uses `.cfg` file as the configuration of Neural Network. In the `.cfg` file, the dilated convolutional layer is configured with:
```
[dilated_convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky
dilate_rate=1
```
Here's a demonstration of dilated convolution and dilate rate:
![window]{./img/dconv_demo.PNG}
Dilated convolution expands its convolutional kernel according to dilate rate. In this demonstration, dilate rate = 1. If dilate rate = 0, dilated convolution is simply normal convolution.

### CSRNet composed with this implementation training result
![window]{./img/CSRNet_density_0.png}
