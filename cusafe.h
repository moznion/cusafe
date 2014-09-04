#ifndef INCLUDE_CUSAFE_H_
#define INCLUDE_CUSAFE_H_

#include <stdio.h>

void cudaSetDeviceSafe(int device_id);

void cudaMallocSafe(void* ptr, int data_size);

void cudaMallocHostSafe(void* ptr, int data_size);

void cudaMemcpySafe(void* dst, const void* src, size_t data_size, cudaMemcpyKind kind);

void cudaMemcpyPeerSafe(void* dst, int dst_device, const void* src, int src_device, size_t data_size);

void cudaFreeSafe(void* ptr);

void cudaFreeHostSafe(void* ptr);

#endif
