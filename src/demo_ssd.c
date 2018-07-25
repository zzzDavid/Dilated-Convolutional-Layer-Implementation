//
//  demo_ssd.c
//  darknet-xcode
//
//  Created by Tony on 2017/8/21.
//  Copyright © 2017年 tony. All rights reserved.
//

#include "demo_ssd.h"
#include "network.h"
#include "detection_output_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include <sys/time.h>

#define DEMO_SSD 1

#ifdef OPENCV

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static float **probs;
static ScoreLabelIndex *sli;
static box_b *boxes;
static network net;
static image buff [3];
static image buff_letter[3];
static int buff_index = 0;
static CvCapture * cap;
static IplImage  * ipl;
static float fps = 0;
static float demo_thresh = 0;
static float demo_hier = .5;
static int running = 0;

static int demo_delay = 0;
static int demo_frame = 3;
static int demo_detections = 0;
static float **predictions;
static int demo_index = 0;
static int demo_done = 0;
static float *last_avg2;
static float *last_avg;
static float *avg;


extern double demo_time;

double ssd_get_wall_time()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

void *ssd_detect_in_thread(void *ptr)
{
    running = 1;
    float nms = .4;
    
    layer l = net.layers[net.n-1];
    float *X = buff_letter[(buff_index+2)%3].data;
    demo_time = ssd_get_wall_time();
    float *prediction = network_predict(net, X);
    fps = 1./(ssd_get_wall_time() - demo_time);
    memcpy(predictions[demo_index], prediction, l.outputs*sizeof(float));
    mean_arrays(predictions, demo_frame, l.outputs, avg);
    l.output = last_avg2;
    if(demo_delay == 0) l.output = avg;
    
    get_ssd_detection_boxes(l, demo_thresh, probs, boxes);
    
    if (nms > 0) do_nms_obj_ssd(boxes, probs, l.num_priors, l.classes, nms, sli, l.top_k, l.keep_top_k, l.eta);
    
    
    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.1f\n",fps);
    printf("Objects:\n\n");
    image display = buff[(buff_index+2) % 3];
    draw_ssd_detections(display, demo_detections, demo_thresh, boxes, sli, demo_names, demo_alphabet, demo_classes);
    

    demo_index = (demo_index + 1)%demo_frame;
    running = 0;
    return 0;
    
    
    
    /*
    running = 1;
//    float nms = .4;
    
    layer l = net.layers[net.n-1];
    float *X = buff_letter[(buff_index+2)%3].data;
    network_predict(net, X);
    
    memcpy(pred_boxes[demo_index], l.all_decode_bboxes, 4 * l.num_priors * l.num_classes *sizeof(float));
    memcpy(pred_slis[demo_index], l.kept_sli, 3 * l.total_num*sizeof(float));
    
    //mean_boxes_float(pred_boxes, demo_frame, l.num_priors * l.num_classes * 4, avg);
    
	//mean_arrays(pred_slis, demo_frame, l.total_num*3, avg);
    
    l.all_decode_bboxes = last_avg2;
    l.kept_sli = last_sli_avg2;
    if(demo_delay == 0) 
    {
		l.all_decode_bboxes = avg;
		l.kept_sli = sli_avg;
	}
    
    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.1f\n",fps);
    printf("Objects:\n\n");
//    image display = buff[(buff_index+2) % 3];
    image display = buff[0];
    draw_ssd_detections_data(display, l.total_num, demo_thresh, l.all_decode_bboxes, l.kept_sli, demo_names, demo_alphabet, demo_classes);//(display, l.total_num, demo_thresh, l.all_decode_bboxes, l.kept_sli, demo_names, demo_alphabet, demo_classes);
    
    
    demo_index = (demo_index + 1)%demo_frame;
    running = 0;
    return 0;
     */
}

void *ssd_fetch_in_thread(void *ptr)
{
    int status = fill_image_from_stream(cap, buff[buff_index]);
    buff_letter[buff_index] = resize_image(buff[buff_index], net.w, net.h);
//    letterbox_image_into(buff[buff_index], net.w, net.h, buff_letter[buff_index]);
    if(status == 0) demo_done = 1;
    return 0;
}

void *ssd_display_in_thread(void *ptr)
{
    show_image_cv(buff[(buff_index + 1)%3], "Demo", ipl);
    int c = cvWaitKey(1);
    if (c != -1) c = c%256;
    if (c == 10){
        if(demo_delay == 0) demo_delay = 60;
        else if(demo_delay == 5) demo_delay = 0;
        else if(demo_delay == 60) demo_delay = 5;
        else demo_delay = 0;
    } else if (c == 27) {
        demo_done = 1;
        return 0;
    } else if (c == 82) {
        demo_thresh += .02;
    } else if (c == 84) {
        demo_thresh -= .02;
        if(demo_thresh <= .02) demo_thresh = .02;
    } else if (c == 83) {
        demo_hier += .02;
    } else if (c == 81) {
        demo_hier -= .02;
        if(demo_hier <= .0) demo_hier = .0;
    }
    return 0;
}

void *ssd_display_loop(void *ptr)
{
    while(1){
        ssd_display_in_thread(0);
    }
}

void *ssd_detect_loop(void *ptr)
{
    while(1){
        ssd_detect_in_thread(0);
    }
}

void ssd_demo(char *cfgfile,
              char *weightfile,
              float thresh,
              int cam_index,
              const char *filename,
              char **names,
              int classes,
              int delay,
              char *prefix,
              int avg_frames,
              float hier,
              int w,
              int h,
              int frames,
              int fullscreen)
{
    demo_delay = delay;
    demo_frame = avg_frames;
    predictions = calloc(demo_frame, sizeof(float *));
    image **alphabet = load_alphabet();
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    demo_hier = hier;
    printf("Demo\n");
    net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    pthread_t detect_thread;
    pthread_t fetch_thread;
    
    srand(2222222);
    
    if(filename){
        printf("camera index %d\n", cam_index);
        printf("video file: %s\n", filename);
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);
        
        if(w){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
        }
        if(h){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
        }
        if(frames){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FPS, frames);
        }
    }
    
    if(!cap) error("Couldn't connect to webcam.\n");
    
    layer l = net.layers[net.n-1];
    demo_detections = l.n*l.w*l.h;
    int j;
    
        
    sli = (ScoreLabelIndex *)calloc(l.num_classes * l.num_priors, sizeof(ScoreLabelIndex));
    avg = (float *) calloc(l.outputs, sizeof(float));
    last_avg  = (float *) calloc(l.outputs, sizeof(float));
    last_avg2 = (float *) calloc(l.outputs, sizeof(float));
    for(j = 0; j < demo_frame; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));
    
    boxes = (box_b *)calloc(l.w*l.h*l.n, sizeof(box_b));
    probs = (float **)calloc(l.num_classes, sizeof(float *));
    for(j = 0; j < l.num_classes; ++j) probs[j] = (float *)calloc(l.num_priors, sizeof(float));
    
    buff[0] = get_image_from_stream(cap);
    buff[1] = copy_image(buff[0]);
    buff[2] = copy_image(buff[0]);
    buff_letter[0] = resize_image(buff[0], net.w, net.h);
    buff_letter[1] = resize_image(buff[0], net.w, net.h);
    buff_letter[2] = resize_image(buff[0], net.w, net.h);
    ipl = cvCreateImage(cvSize(buff[0].w,buff[0].h), IPL_DEPTH_8U, buff[0].c);
    
    int count = 0;
    if(!prefix){
        cvNamedWindow("Demo", CV_WINDOW_NORMAL);
        if(fullscreen){
            cvSetWindowProperty("Demo", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
        } else {
            cvMoveWindow("Demo", 0, 0);
            cvResizeWindow("Demo", 1352, 1013);
        }
    }
    
    demo_time = ssd_get_wall_time();
    
    while(!demo_done){
        buff_index = (buff_index + 1) %3;
        //if(pthread_create(&fetch_thread, 0, ssd_fetch_in_thread, 0)) error("Thread creation failed");
        //if(pthread_create(&detect_thread, 0, ssd_detect_in_thread, 0)) error("Thread creation failed");
        ssd_fetch_in_thread(NULL);
        ssd_detect_in_thread(NULL);
        if(!prefix){
            if(count % (demo_delay+1) == 0){

                float *swap = last_avg;
                last_avg  = last_avg2;
                last_avg2 = swap;
                memcpy(last_avg, avg, l.outputs *sizeof(float));
            }
            ssd_display_in_thread(0);
        }else{
            char name[256];
            sprintf(name, "%s_%08d", prefix, count);
            save_image(buff[(buff_index + 1)%3], name);
        }
        //pthread_join(fetch_thread, 0);
        //pthread_join(detect_thread, 0);
        ++count;
    }
}
#else
void ssd_demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg, float hier, int w, int h, int frames, int fullscreen)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif

