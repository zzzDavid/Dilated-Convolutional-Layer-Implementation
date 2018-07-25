#include "queue.h"
#include <stdlib.h>
#include <time.h>
#include <errno.h>
Queue *CreateNewQueue(int queueSize) {
	Queue* qPtr = (Queue *)calloc(1, sizeof(Queue));

	if(!qPtr)
		goto Err1;
	qPtr->mutex = (pthread_mutex_t *)calloc(1, sizeof(pthread_mutex_t));
	qPtr->cond = (pthread_cond_t *)calloc(1, sizeof(pthread_cond_t));
	qPtr->maxQueueSize = queueSize;
	qPtr->payload = (void **)calloc(1, queueSize * sizeof(void *));
	if(!qPtr->payload)
		goto Err2;
	if(0 != pthread_mutex_init(qPtr->mutex, NULL))
		goto Err3;
	pthread_condattr_t cattr;
	pthread_condattr_init(&cattr);
	pthread_condattr_setclock(&cattr, CLOCK_MONOTONIC);
	if(0 != pthread_cond_init(qPtr->cond, &cattr))
		goto Err4;
	pthread_condattr_destroy(&cattr);
	return qPtr;
Err4:
	pthread_mutex_destroy(qPtr->mutex);
Err3:
	free(qPtr->payload);
Err2:
	free(qPtr);
Err1:
	return NULL;

}

void setQueueElement(Queue *q, void *payload, int index) {
	q->payload[index] = payload;
}


void DestoryQueue(Queue **qPtr) {
	if(qPtr && *qPtr) {
		Queue *doom = *qPtr;
		pthread_mutex_destroy(doom->mutex);
		pthread_cond_destroy(doom->cond);
		free(doom->mutex);
		free(doom->cond);
		free(doom->payload);
		free(doom);
		*qPtr = NULL;
	}
}
static 	const int timeStepInNanoSeconds = 100000; // 100us

void* peekQueueWritable(Queue *q, int *exit) {
	struct timespec tv;
	pthread_mutex_lock(q->mutex);
	while(q->size == q->maxQueueSize
			&& !*exit) {
		clock_gettime(CLOCK_MONOTONIC, &tv);
		tv.tv_nsec += timeStepInNanoSeconds;
		pthread_cond_timedwait(q->cond, q->mutex, &tv);

	}
	pthread_mutex_unlock(q->mutex);

	return q->payload[q->wIndex];
}

void pushQueue(Queue *q) {
	pthread_mutex_lock(q->mutex);
	if(++q->wIndex == q->maxQueueSize)
		q->wIndex = 0;
	q->size++;
	pthread_cond_signal(q->cond);
	pthread_mutex_unlock(q->mutex);

}

void* peekQueueReadable(Queue *q, int *exit) {
	struct timespec tv;
	pthread_mutex_lock(q->mutex);
	 while (q->size == 0
				&& !*exit) {
		clock_gettime(CLOCK_MONOTONIC, &tv);
		tv.tv_nsec += timeStepInNanoSeconds;
		pthread_cond_timedwait(q->cond, q->mutex, &tv);
	}
	pthread_mutex_unlock(q->mutex);
	return q->payload[q->rIndex];
}


void popQueue(Queue *q) {
	pthread_mutex_lock(q->mutex);
	if(q->size > 0) {
		if(++q->rIndex == q->maxQueueSize)
			q->rIndex = 0;
		q->size--;
	}
	pthread_cond_signal(q->cond);
	pthread_mutex_unlock(q->mutex);
}
