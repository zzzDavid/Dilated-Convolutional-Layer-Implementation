//
//  hash_map.h
//  darknet-xcode
//
//  Created by Tony on 2017/7/27.
//  Copyright © 2017年 tony. All rights reserved.
//

#ifndef hash_map_h
#define hash_map_h

#include <stdio.h>
#include "unistd.h"

#define HASHMAP_SIZE 1024

typedef struct _hash_map_t* hash_map;

typedef void(*pfcb_hmap_value_free) (void* value);

extern void hmap_create(hash_map * hmap, int size);

extern void hmap_destroy(hash_map hmap, pfcb_hmap_value_free);

extern void hmap_insert(hash_map hmap, const char* key, int key_len, void* value);

extern void* hmap_search(hash_map hmap, const char *key);

#endif /* hash_map_h */
