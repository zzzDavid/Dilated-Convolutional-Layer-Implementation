//
//  detection_output_layer.c
//  darknet-xcode
//
//  Created by Tony on 2017/7/27.
//  Copyright © 2017年 tony. All rights reserved.
//

#include "detection_output_layer.h"
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include "blas.h"
#include "darknet.h"



detection_output_layer make_detection_output_layer(network net,
                                                   int *input_layers,
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
                                                   float eta)
{
    detection_output_layer l = {0};
    l.type = DETECTION_OUTPUT;
    l.batch = net.batch;
    l.num_classes = num_classes;
    l.classes = num_classes;
    l.share_location = share_location;
    l.background_label_id = background_label_id;
    l.nms_threshold = nms_threshold;
    l.top_k = top_k;
    l.keep_top_k = keep_top_k;
    l.confidence_threshold = confidence_threshold;
    l.code_type = code_type;
    l.eta = eta;
    l.variance_encoded_in_target = variance_encoded_in_target;
    
    l.forward = forward_detection_output_layer;
    l.input_layers = input_layers;
    l.num_input_layers = num_input_layers;
    
    l.outputs = l.keep_top_k;
    
    l.total_num = l.batch * l.keep_top_k;// 输出box数
    
    int i = 0;
    int has_prior = 0;
    int has_loc = 0;
    int has_conf = 0;
    
    int loc_data_len = 0;
    int priorbox_data_len = 0;
    int conf_data_len = 0;
    
    for (i = 0; i < l.num_input_layers; ++i) {
        int index = l.input_layers[i];
        layer input_layer = net.layers[index];
        if (input_layer.det_in_layer_type == Loc && has_loc == 0) {
            loc_data_len = input_layer.outputs * input_layer.batch;
            l.loc_data = (float*)malloc(loc_data_len * sizeof(float));
            has_loc = 1;
        } else if (input_layer.det_in_layer_type == Confidence && has_conf == 0) {

            conf_data_len = input_layer.outputs * input_layer.batch;
            l.conf_data = (float*)malloc(conf_data_len * sizeof(float));
            has_conf = 1;
        } else if (input_layer.det_in_layer_type == PriorBox && has_prior == 0) {
            has_prior = 1;
            l.num_priors = input_layer.out_h * input_layer.out_w / 4;
            
            priorbox_data_len = input_layer.outputs * input_layer.batch;
            l.prior_data = (float*)malloc(priorbox_data_len * sizeof(float));
        }
        
    }
    
//    l.all_decode_bboxes = (float*)malloc(sizeof(float) * l.batch * l.num_priors * 4);
//    l.kept_sli = (float*)malloc(sizeof(float) * l.total_num * 3);

    l.w = 1;
    l.h = 1;
    l.n = l.num_priors;
    l.outputs = l.num_priors * (4 + l.num_classes);
    l.output = (float*)malloc(sizeof(float) * l.batch * l.outputs);
    
    
    
#ifdef GPU
    l.forward_gpu = forward_detection_output_layer_gpu;
    l.loc_data_gpu = cuda_make_array(l.loc_data, loc_data_len);
    l.prior_data_gpu = cuda_make_array(l.prior_data, priorbox_data_len);
    l.conf_data_gpu = cuda_make_array(l.conf_data, conf_data_len);
//    l.all_box_data_gpu = cuda_make_array(l.all_decode_bboxes, l.batch * l.num_priors * 4);
//    l.kept_sli_gpu = cuda_make_array(l.kept_sli, l.total_num * 3);
    l.output_gpu = cuda_make_array(l.output, l.batch * l.outputs);
#endif
    printf("success make detection output layer.\n");
    return l;
}

void permute_conf_data(const float* conf_data,
                         const int num,
                         const int num_preds_per_class,
                         const int num_classes,
                         float* conf_preds)
{
    int i, p, c;
    for (i = 0; i < num; ++i) {
        for (c = 0; c < num_classes; ++c) {
            for (p = 0; p < num_preds_per_class; ++p) {
                int des_index = (i * num_classes + c) * num_preds_per_class + p;
                int src_index = (i * num_preds_per_class + p) * num_classes + c;
                conf_preds[des_index] = conf_data[src_index];
            }
        }
    }
}


//输入 prior_data, num_priors
//输出 prior_bboxes, prior_variances
void getPriorBBoxes(const float *prior_data, const int num_priors, NormalizedBox *prior_bboxes, float* prior_variances)
{
//    printf("prior box: \n");
//    printf("    xmin    ymin    xmax    ymax    size    \n");
    int i, j;
    for (i=0; i<num_priors; ++i) {
        int start_idx = i * 4;
        NormalizedBox bbox;
        bbox.xmin = prior_data[start_idx];
        bbox.ymin = prior_data[start_idx + 1];
        bbox.xmax = prior_data[start_idx + 2];
        bbox.ymax = prior_data[start_idx + 3];
        bbox.size = BBoxSize(bbox, true);
//        float bbox_size = sizeof(NormalizedBox);
//        bbox.size = bbox_size;
        prior_bboxes[i] = bbox;
//        printf("  %6f    %6f    %6f    %6f     %6f\n", bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax, bbox.size);
    }
    //printf("variance: \n");
    
    for (i=0; i<num_priors; ++i) {
        int start_idx = (num_priors + i) * 4;
        for (j=0; j<4; ++j) {
            prior_variances[i*4 + j] = prior_data[start_idx + j];
//            printf("   %6f\n", prior_variances[i * 4 + j]);
        }
//        printf("\n");
    }
}





void check_gt(float x, const float y)
{
    x = (x - y > 1e-6) ? x : y;
}

void check_le(float x, const float y)
{
    x = (y - x > 1e-6 ) ? x : y;
}


void decode_box_ssd( float *prior_bbox,
                     float * prior_variance,
                    const Code_Type code_type,
                    const int variance_encoded_in_target,
                    const int clip_bbox,
                    const float *bbox,
                    float *decode_bbox)
{
	printf("--- input box: %f, %f, %f, %f  \n", bbox[0], bbox[1], bbox[2], bbox[3]);
	printf("--- prior box: %f, %f, %f, %f  \n", prior_bbox[0], prior_bbox[1], prior_bbox[2], prior_bbox[3]);
	printf("--- prior_variance: %f, %f, %f, %f  \n", prior_variance[0], prior_variance[1], prior_variance[2], prior_variance[3]);
	
    if (code_type == PriorBoxParameter_CodeType_CORNER) {
        if (variance_encoded_in_target) {
            decode_bbox[0] = prior_bbox[0] + bbox[0];
            decode_bbox[1] = prior_bbox[1] + bbox[1];
            decode_bbox[2] = prior_bbox[2] + bbox[2];
            decode_bbox[3] = prior_bbox[3] + bbox[3];
        } else {
            decode_bbox[0] = prior_bbox[0] + prior_variance[0] * bbox[0];
            decode_bbox[1] = prior_bbox[1] + prior_variance[1] * bbox[1];
            decode_bbox[2] = prior_bbox[2] + prior_variance[2] * bbox[2];
            decode_bbox[3] = prior_bbox[3] + prior_variance[3] * bbox[3];
        }
    } else if(code_type == PriorBoxParameter_CodeType_CENTER_SIZE) {
		float prior_width = prior_bbox[2] - prior_bbox[0];
        check_gt(prior_width, 0.f);
        
        float prior_height = prior_bbox[3] - prior_bbox[1];
        check_gt(prior_height, 0.f);
        
        float prior_center_x = (prior_bbox[0] + prior_bbox[2]) / 2.f;
        float prior_center_y = (prior_bbox[1] + prior_bbox[3]) / 2.f;
        
        float decode_bbox_center_x,decode_bbox_center_y;
        float decode_bbox_width,decode_bbox_height;
        if (variance_encoded_in_target) {
            decode_bbox_center_x = bbox[0] * prior_width + prior_center_x;
            decode_bbox_center_y = bbox[1] * prior_height + prior_center_y;
            decode_bbox_width = expf(bbox[2]) * prior_width;
            decode_bbox_height= expf(bbox[3]) * prior_height;
        } else {
            decode_bbox_center_x = prior_variance[0] * bbox[0] * prior_width + prior_center_x;
            decode_bbox_center_y = prior_variance[1] * bbox[1] * prior_height + prior_center_y;
            decode_bbox_width   = expf(prior_variance[2] * bbox[2]) * prior_width;
            decode_bbox_height  = expf(prior_variance[3] * bbox[3]) * prior_height;
        }
        decode_bbox[0] = decode_bbox_center_x - decode_bbox_width / 2.f;
        decode_bbox[1] = decode_bbox_center_y - decode_bbox_height / 2.f;
        decode_bbox[2] = decode_bbox_center_x + decode_bbox_width / 2.f;
        decode_bbox[3] = decode_bbox_center_y + decode_bbox_height / 2.f;
    } else if (code_type == PriorBoxParameter_CodeType_CORNER_SIZE) {
        float prior_width = prior_bbox[2] - prior_bbox[0];
        check_gt(prior_width, 0.f);
        
        float prior_height = prior_bbox[3] - prior_bbox[1];
        check_gt(prior_height, 0.f);
        
        if (variance_encoded_in_target) {
            decode_bbox[0] = prior_bbox[0] + bbox[0] * prior_width;
            decode_bbox[1] = prior_bbox[1] + bbox[1] * prior_height;
            decode_bbox[2] = prior_bbox[2] + bbox[2] * prior_width;
            decode_bbox[3] = prior_bbox[3] + bbox[3]* prior_height;
        } else {
            decode_bbox[0] = prior_bbox[0] + prior_variance[0] * bbox[0] * prior_width;
            decode_bbox[1] = prior_bbox[1] + prior_variance[1] * bbox[1] * prior_height;
            decode_bbox[2] = prior_bbox[2] + prior_variance[2] * bbox[2] * prior_width;
            decode_bbox[3] = prior_bbox[3] + prior_variance[3] * bbox[3] * prior_height;
        }
    } else {
        printf("unkown loclosstype");
    }
    printf("--- decoded box: %f, %f, %f, %f  \n", decode_bbox[0], decode_bbox[1], decode_bbox[2], decode_bbox[3]);
//    float bbox_size = fBBoxSize(decode_bbox, true);// normalize 参数不确定;
    
    if (clip_bbox) {
        fClipBbox(decode_bbox, decode_bbox);
    }
    
}


void decodeBBox(const NormalizedBox prior_bbox,
                const float* prior_variance,
                const Code_Type code_type,
                const int variance_encoded_in_target,
                const int clip_bbox,
                const NormalizedBox bbox,
                NormalizedBox *decode_bbox)
{
    if (code_type == PriorBoxParameter_CodeType_CORNER) {
        if (variance_encoded_in_target) {
            decode_bbox->xmin = prior_bbox.xmin + bbox.xmin;
            decode_bbox->ymin = prior_bbox.ymin + bbox.ymin;
            decode_bbox->xmax = prior_bbox.xmax + bbox.xmax;
            decode_bbox->ymax = prior_bbox.ymax + bbox.ymax;
        } else {
            decode_bbox->xmin = prior_bbox.xmin + prior_variance[0] * bbox.xmin;
            decode_bbox->ymin = prior_bbox.ymin + prior_variance[1] * bbox.ymin;
            decode_bbox->xmax = prior_bbox.xmax + prior_variance[2] * bbox.xmax;
            decode_bbox->ymax = prior_bbox.ymax + prior_variance[3] * bbox.ymax;
        }
    } else if(code_type == PriorBoxParameter_CodeType_CENTER_SIZE) {
        float prior_width = prior_bbox.xmax - prior_bbox.xmin;
        check_gt(prior_width, 0.f);
        
        float prior_height = prior_bbox.ymax - prior_bbox.ymin;
        check_gt(prior_height, 0.f);
        
        float prior_center_x = (prior_bbox.xmin + prior_bbox.xmax) / 2.f;
        float prior_center_y = (prior_bbox.ymin + prior_bbox.ymax) / 2.f;
        
        float decode_bbox_center_x,decode_bbox_center_y;
        float decode_bbox_width,decode_bbox_height;
        if (variance_encoded_in_target) {
            decode_bbox_center_x = bbox.xmin * prior_width + prior_center_x;
            decode_bbox_center_y = bbox.ymin * prior_height + prior_center_y;
            decode_bbox_width = expf(bbox.xmax) * prior_width;
            decode_bbox_height= expf(bbox.ymax) * prior_height;
        } else {
            decode_bbox_center_x = prior_variance[0] * bbox.xmin * prior_width + prior_center_x;
            decode_bbox_center_y = prior_variance[1] * bbox.ymin * prior_height + prior_center_y;
            decode_bbox_width   = expf(prior_variance[2] * bbox.xmax) * prior_width;
            decode_bbox_height  = expf(prior_variance[3] * bbox.ymax) * prior_height;
        }
        decode_bbox->xmin = decode_bbox_center_x - decode_bbox_width / 2.f;
        decode_bbox->ymin = decode_bbox_center_y - decode_bbox_height / 2.f;
        decode_bbox->xmax = decode_bbox_center_x + decode_bbox_width / 2.f;
        decode_bbox->ymax = decode_bbox_center_y + decode_bbox_height / 2.f;
    } else if (code_type == PriorBoxParameter_CodeType_CORNER_SIZE) {
        float prior_width = prior_bbox.xmax - prior_bbox.xmin;
        check_gt(prior_width, 0.f);

        float prior_height = prior_bbox.ymax - prior_bbox.ymin;
        check_gt(prior_height, 0.f);

        if (variance_encoded_in_target) {
            decode_bbox->xmin = prior_bbox.xmin + bbox.xmin * prior_width;
            decode_bbox->ymin = prior_bbox.ymin + bbox.ymin * prior_height;
            decode_bbox->xmax = prior_bbox.xmax + bbox.xmax * prior_width;
            decode_bbox->ymax = prior_bbox.ymax + bbox.ymax * prior_height;
        } else {
            decode_bbox->xmin = prior_bbox.xmin + prior_variance[0] * bbox.xmin * prior_width;
            decode_bbox->ymin = prior_bbox.ymin + prior_variance[1] * bbox.ymin * prior_height;
            decode_bbox->xmax = prior_bbox.xmax + prior_variance[2] * bbox.xmax * prior_width;
            decode_bbox->ymax = prior_bbox.ymax + prior_variance[3] * bbox.ymax * prior_height;
        }
    } else {
        printf("unkown loclosstype");
    }
    
    float bbox_size = BBoxSize(*decode_bbox, true);// normalize 参数不确定;
    
    decode_bbox->size = bbox_size;
    
    
    if (clip_bbox) {
        ClipBbox(*decode_bbox, decode_bbox);
    }
    
    
}

void decode_all_box(float *loc_data,
                        float *priorbox_data,
                         const int num_priors,
                         const int num,
                         const int share_location,
                         const int num_loc_classes,
                         const int background_label_id,
                         const int code_type,
                         const int variance_encoded_in_target,
                         const int clip,
                         float* all_decode_bbox_data)
{
    int i, j;
    
    float *prior_variances = priorbox_data + num_priors * 4;
    
    for (i = 0; i < num; i++) {
        for (j = 0; j < num_priors; ++j) {
            int offset = (i * num_priors + j) * 4;
            decode_box_ssd(priorbox_data + 4 * j,
                       prior_variances + i * 4,
                       code_type,
                       variance_encoded_in_target,
                       clip,
                       loc_data + offset,
                       all_decode_bbox_data + offset);
        }
    }
}





void forward_detection_cpu(layer l, network net)
{
    int num_loc_classes = l.share_location ? 1: l.num_classes;
    
    float *all_conf_scores = l.output + l.num_priors * 4;
//    memcpy(all_conf_scores, l.conf_data, l.num_classes * l.num_priors);
    permute_conf_data(l.conf_data, l.batch, l.num_priors, l.num_classes, all_conf_scores);
    printf("num_priors: %d, share_location:%d, num_loc_classes:%d, back_id:%d, code_type:%d, varean_encode_in_target:%d", l.num_priors, l.share_location, num_loc_classes, l.background_label_id, l.code_type, l.variance_encoded_in_target);
    decode_all_box(l.loc_data,
                   l.prior_data,
                   l.num_priors,
                   l.batch, l.share_location,
                   num_loc_classes,
                   l.background_label_id,
                   l.code_type,
                   l.variance_encoded_in_target,
                   0,
                   l.output);
}
/*
void forward_detection_cpu(layer l, network net)
{
    
    int i,c;
    int num_loc_classes = l.share_location ? 1 : l.num_classes;
    
    float *all_conf_scores = (float*)malloc(l.batch * l.num_priors * l.num_classes * sizeof(float));
    permute_conf_data(l.conf_data, l.batch, l.num_priors, l.num_classes, all_conf_scores);
    
    decode_all_box(l.loc_data,
                   l.prior_data,
                   l.num_priors,
                   l.batch, l.share_location,
                   num_loc_classes,
                   l.background_label_id,
                   l.code_type,
                   l.variance_encoded_in_target,
                   0,
                   l.all_decode_bboxes);
    
    int *num_det_arr = (int*)malloc(l.batch *l.num_classes * sizeof(int));
    int k;
    l.total_num = l.batch * l.keep_top_k;
    //    int total_num = l.batch * l.keep_top_k;// 输出box数
    //    ScoreLabelIndex *kept_sli = (ScoreLabelIndex*)malloc(sizeof(ScoreLabelIndex) * total_num);
    
    int actual_kept_num = 0;
    for (i = 0; i < l.batch; ++i) {
        
        int num_det = 0;
        float *conf_scores = all_conf_scores + i * l.num_priors * l.num_classes;
        float *bboxes = l.all_decode_bboxes + i * l.num_priors * 4;
        float *kept_sli_per_batch = (float*)malloc(l.num_classes * l.num_priors * 3 * sizeof(float));
        int kept_num_per_img = 0;
        
        float *sli_per_class = (float*)malloc(l.num_priors * 3 * sizeof(float));
        
        for (c = 0; c < l.num_classes; ++c) {
            if (c == l.background_label_id) {
                continue;
            }
            // 每一类在每个框上的置信度
            float *scores = conf_scores + c * l.num_priors;
            
            int n = 0;
            
            // make pair score label and index
            
            
            for (k = 0; k < l.num_priors; ++k) {
                if (scores[k] > l.confidence_threshold) {
                    int idx = n * 3;
                    sli_per_class[idx] = scores[k];
                    sli_per_class[idx + 1] = k;
                    sli_per_class[idx + 2] = c;
                    n++;
                }
            }
            
            
            bin_sort_group_float(sli_per_class, 3, n);
            
            //top k
            //qsort(sli_per_class, n, sizeof(ScoreLabelIndex), cmp_descend2);
            
            int m = n > l.top_k ? l.top_k : n;
            int *tmp_sli = (int *)malloc(m * 3 * sizeof(int));
            
            for (k = 0; k < m * 3; k ++) {
                tmp_sli[k] = sli_per_class[k];
            }
            
            //ScoreLabelIndex *kept_sli = (ScoreLabelIndex*)malloc(m * sizeof(ScoreLabelIndex));
            // nms
            float adaptive_threshold = l.nms_threshold;
            int kept_num_per_class = 0;
            for (k = 0; k < m; ++k) {
                int idx = sli_per_class[k * 3 + 1];
                bool keep = true;
                int ii = 0;
                for (ii=0; ii < kept_num_per_class; ++ii) {
                    if (keep) {
                        const int kept_index = tmp_sli[ii * 3 + 1];
                        float overlap = fJaccardOverlap(bboxes+idx*4, bboxes+kept_index*4);
                        keep = overlap <= adaptive_threshold;
                    } else {
                        break;
                    }
                }
                if (keep) {
                    //printf("nms output score: %f  label: %d  index %d \n ", sli_per_class[k].score, sli_per_class[k].label, sli_per_class[k].index);
                    tmp_sli[kept_num_per_class * 3 ] = sli_per_class[k * 3];
                    tmp_sli[kept_num_per_class * 3 +1] = sli_per_class[k * 3 + 1];
                    tmp_sli[kept_num_per_class * 3 +2] = sli_per_class[k * 3 + 2];
                    kept_sli_per_batch[kept_num_per_img * 3 ] = sli_per_class[k * 3];
                    kept_sli_per_batch[kept_num_per_img * 3 +1] = sli_per_class[k * 3 + 1];
                    kept_sli_per_batch[kept_num_per_img * 3 +2] = sli_per_class[k * 3 + 2];
                    kept_num_per_img ++;
                    kept_num_per_class ++;
                }
                
                if (keep && l.eta < 1 && adaptive_threshold > 0.5) {
                    adaptive_threshold *= l.eta;
                }
            }
            
            free(tmp_sli);
            tmp_sli = NULL;
            
            num_det_arr[i * l.num_classes + c] = kept_num_per_class;
            num_det += kept_num_per_class;
        }
        
        free(sli_per_class);
        sli_per_class = NULL;
        
        if ( l.keep_top_k > -1 && num_det > l.keep_top_k) {
            //qsort(kept_sli_per_batch, num_det, sizeof(ScoreLabelIndex), cmp_descend2);
            
            bin_sort_group_float(kept_sli_per_batch, 3, num_det);
            
            for (k = 0; k < l.keep_top_k; ++k) {
                l.kept_sli[(k + actual_kept_num)*3] = kept_sli_per_batch[k*3];
                l.kept_sli[(k + actual_kept_num)*3 + 1] = kept_sli_per_batch[k*3 + 1];
                l.kept_sli[(k + actual_kept_num)*3 + 2] = kept_sli_per_batch[k*3 + 2];
            }
            actual_kept_num += l.keep_top_k;
        } else {
            //qsort(kept_sli_per_batch, num_det, sizeof(ScoreLabelIndex), cmp_descend2);
            
            bin_sort_group_float(kept_sli_per_batch, 3, num_det);
            
            for (k = 0; k < num_det; ++k) {
                l.kept_sli[(k + actual_kept_num)*3] = kept_sli_per_batch[k*3];
                l.kept_sli[(k + actual_kept_num)*3 + 1] = kept_sli_per_batch[k*3+1];
                l.kept_sli[(k + actual_kept_num)*3 + 2] = kept_sli_per_batch[k*3+2];
            }
            actual_kept_num += num_det;
        }
        
        free(kept_sli_per_batch);
        kept_sli_per_batch = NULL;
    }
    l.total_num = actual_kept_num;
}
*/


void forward_detection_output_layer(layer l, network net)
{
    int i = 0, j = 0;
    
    int has_prior = 0;
    int has_loc = 0;
    int has_conf = 0;
    fprintf(stderr, "detection_output_layer forward.\n");
    for (i = 0; i < l.num_input_layers; ++i) {
        int index = l.input_layers[i];
        layer input_layer = net.layers[index];
        if (input_layer.det_in_layer_type == Loc && has_loc == 0) {
			printf("--Loc:");
			for (j = 0; j < 10; j ++)
				printf(" %f ", input_layer.output[10+j]);
			printf("\n");
            memcpy(l.loc_data, input_layer.output, input_layer.outputs * input_layer.batch * sizeof(float));
            has_loc = 1;
        } else if (input_layer.det_in_layer_type == Confidence && has_conf == 0) {
			printf("--Conf:");
			for (j = 0; j < 10; j ++)
				printf(" %f ", input_layer.output[10+j]);
			printf("\n");
            memcpy(l.conf_data, input_layer.output, input_layer.outputs * input_layer.batch * sizeof(float));
            has_conf = 1;
        } else if (input_layer.det_in_layer_type == PriorBox && has_prior == 0) {
			printf("--PriorBox:");
			for (j = 0; j < 10; j ++)
				printf(" %f ", input_layer.output[10+j]);
			printf("\n");
            has_prior = 1;
            l.num_priors = input_layer.out_h * input_layer.out_w / 4;
            memcpy(l.prior_data, input_layer.output, input_layer.outputs * input_layer.batch * sizeof(float));
        }
    }

    forward_detection_cpu(l, net);
    
}





#ifdef GPU


void forward_detection_gpu(layer l, network net)
{
    int num_loc_classes = l.share_location ? 1: l.num_classes;
    
//    float *all_conf_scores = l.output_gpu + l.num_priors * 4;
    //    memcpy(all_conf_scores, l.conf_data, l.num_classes * l.num_priors);
//    permute_conf_data(l.conf_data, l.batch, l.num_priors, l.num_classes, all_conf_scores);
    
    decode_all_box_gpu(l.loc_data_gpu,
                   l.prior_data_gpu,
                   l.num_priors,
                   l.batch, l.share_location,
                   num_loc_classes,
                   l.background_label_id,
                   l.code_type,
                   l.variance_encoded_in_target,
                   0,
                   l.output_gpu);
}


void forward_detection_output_layer_gpu(layer l, network net)
{
    int i = 0, c = 0, j = 0;
    int num_loc_classes = l.share_location ? 1 : l.num_classes;
    
    int has_prior = 0;
    int has_loc = 0;
    int has_conf = 0;
    //fprintf(stderr, "detection_output_layer forward.\n");
    for (i = 0; i < l.num_input_layers; ++i) {
        int index = l.input_layers[i];
        layer input_layer = net.layers[index];
        if (input_layer.det_in_layer_type == Loc && has_loc == 0) {
            l.loc_data_gpu = input_layer.output_gpu;
//            cuda_pull_array(l.loc_data_gpu, l.loc_data, input_layer.outputs * input_layer.batch);
            has_loc = 1;
        } else if (input_layer.det_in_layer_type == Confidence && has_conf == 0) {
            has_conf = 1;
//            l.conf_data_gpu = input_layer.output_gpu;
            permute_conf_data_gpu(input_layer.output_gpu, l.batch, l.num_priors, l.num_classes, l.output_gpu + l.num_priors * 4);
//            cuda_pull_array(l.conf_data_gpu, l.conf_data, input_layer.outputs * input_layer.batch);
        } else if (input_layer.det_in_layer_type == PriorBox && has_prior == 0) {
            has_prior = 1;
            l.num_priors = input_layer.out_h * input_layer.out_w / 4;
            l.prior_data_gpu = input_layer.output_gpu;
//            cuda_pull_array(l.prior_data_gpu, l.prior_data, input_layer.outputs * input_layer.batch);
        }
    }
    
    
    forward_detection_gpu(l, net);
    
//    cuda_push_array(l.kept_sli_gpu, l.kept_sli, l.total_num * 3);
//    cuda_push_array(l.all_box_data_gpu, l.all_decode_bboxes, l.batch * l.num_priors * 4);
    
    
    //fprintf(stderr, "detection output: \n");
    //fprintf(stderr, "    total num: %d \n", l.total_num);
    
}

#endif
