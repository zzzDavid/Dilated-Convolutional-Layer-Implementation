#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H
#include "layer.h"
#include "network.h"

typedef layer softmax_layer;

void softmax_array(float *input, int n, float temp, float *output);
//softmax_layer make_softmax_layer(int batch, int inputs, int groups);
softmax_layer make_softmax_layer(int *input_layer_indexes, int classes, int axis, int batch, int c, int h, int w, int inputs, int groups);
void forward_softmax_layer(softmax_layer l, network net);
void backward_softmax_layer(const softmax_layer l, network net);

#ifdef GPU
void pull_softmax_layer_output(const softmax_layer l);
void forward_softmax_layer_gpu(const softmax_layer l, network net);
void backward_softmax_layer_gpu(const softmax_layer l, network net);
#endif

#endif
