/*
 * queue.h
 *
 *  Created on: Mar 22, 2018
 *      Author: wtd
 */

#ifndef QUEUE_H_
#define QUEUE_H_

#include <pthread.h>

typedef struct {
	int maxQueueSize;
	int size;
	void **payload;
	pthread_mutex_t *mutex;
	pthread_cond_t *cond;
	int rIndex;
	int wIndex;
}Queue;

Queue *CreateNewQueue(int maxQueueSize);
void setQueueElement(Queue *q, void *payload, int index);
void DestoryQueue(Queue **qPtr);
void* peekQueueWritable(Queue *q, int *exit);
void* peekQueueReadable(Queue *q, int *exit);
void pushQueue(Queue *q);
void popQueue(Queue *q);
//TODO
//int pushQueueAsync(void *payload, int *exit);
//void *popQueueAsync();



#endif /* QUEUE_H_ */
