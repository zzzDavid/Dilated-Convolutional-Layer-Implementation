//
//  detection_output_layer.h
//  darknet-xcode
//
//  Created by Tony on 2017/7/27.
//  Copyright © 2017年 tony. All rights reserved.
//

#ifndef DETECTION_OUTPUT_LAYER_H
#define DETECTION_OUTPUT_LAYER_H

#include "cuda.h"
#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"

#include <stdio.h>


typedef layer detection_output_layer;

/**
 * @brief Generate the detection output based on location and confidence
 * predictions by doing non maximum suppression.
 *
 * Intended for use with MultiBox detection method.
 *
 * NOTE: does not implement Backwards operation.
 */

void getLocPredictions(const float *loc_data,           // input - location predictions
                       const int num,                   // input - fature map cells num
                       const int num_preds_per_class,   // default boxes number
                       const int num_loc_classes,       // 1
                       const bool share_location,       // true
                       NormalizedBox *loc_preds);       //  output

void getConfidenceScores(const float* conf_data,
                         const int num,
                         const int num_preds_per_class,
                         const int num_classes,
                         float* conf_preds);

void getPriorBBoxes(const float *prior_data, const int num_priors, NormalizedBox *prior_bboxes, float* prior_variances);

float BBoxSize(const NormalizedBox bbox, const bool normalized);

float fBBoxSize(const float *bbox, const bool normalized);

void ClipBbox(const NormalizedBox bbox, NormalizedBox *clipBox);

void decodeBBox(const NormalizedBox prior_bbox,
                const float* prior_variance,
                const Code_Type code_type,
                const int variance_encoded_in_target,
                const int clip_bbox,
                const NormalizedBox bbox,
                NormalizedBox *decode_bbox);

void decodeBBoxesAll(const NormalizedBox* all_loc_preds,
                     const NormalizedBox* prior_bboxes,
                     const int num_priors,
                     const float* prior_variances,
                     const int num,
                     const int share_location,
                     const int num_loc_classes,
                     const int background_label_id,
                     const int code_type,
                     const int variance_encoded_in_target,
                     const int clip,
                     NormalizedBox* all_decode_bboxes);

NormalizedBox intersectBBox(const NormalizedBox bbox1, const NormalizedBox bbox2);

float jaccardOverlap(const NormalizedBox bbox1, NormalizedBox bbox2, const bool normalized);

float fJaccardOverlap(const float *bbox1, const float *bbox2);

int cmp_descend(const void* a, const void* b);

int cmp_descend2(const void* a, const void* b);

int getMaxScoreIndex(const float *scores,
                     const int score_num,
                     const float threshold,
                     ScoreIndex* score_index);

int applyNMS(NormalizedBox* bboxes,
             float* scores,const int score_num,
             const float score_threshold,
             const float nms_threshold,
             const float eta,
             const int top_k,
             int* indices,
             ScoreIndex *topKScoreIndex,
             ScoreIndex *nms_score_index);

void forward_detection_output_layer(layer l, network net);

detection_output_layer make_detection_output_layer(network net,
                                                   int *from_layers_id,
                                                   int num_input_layers,
                                                   int num_classes,
                                                   bool share_location,
                                                   int background_label_id,
                                                   int variance_encoded_in_target,
                                                   float nms_threshold,
                                                   int top_k,
                                                   int keep_top_k,
                                                   Code_Type code_type,
                                                   float confidence_threshold,
                                                   float eta);
#ifdef GPU
void forward_detection_output_layer_gpu(layer l, network net);

#endif
#endif /* DETECTION_OUTPUT_LAYER_H */
