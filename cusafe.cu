#include "cusafe.h"

void cudaSetDeviceSafe(int device_id) {
    cudaError_t err = cudaSetDevice(device_id);

    if (err) {
        fprintf(stderr, "Cannot set a device (device: %d)\n", device_id);
        exit(1);
    }
}

void cudaMallocSafe(void* ptr, int data_size) {
    cudaError_t err = cudaMalloc((void **)ptr, data_size);

    if (err) {
        fprintf(stderr, "Failed to allocate the memory on device\n");
        exit(1);
    }
}

void cudaMallocHostSafe(void* ptr, int data_size) {
    cudaError_t err = cudaMallocHost((void **)ptr, data_size);

    if (err) {
        fprintf(stderr, "Failed to allocate the memory on host\n");
        exit(1);
    }
}

void cudaMemcpySafe(void* dst, const void* src, size_t data_size, cudaMemcpyKind kind) {
    cudaError_t err = cudaMemcpy(dst, src, data_size, kind);

    if (err) {
        fprintf(stderr, "Failed to copy memory (direction: %s)\n", kind);
        exit(1);
    }
}

void cudaMemcpyPeerSafe(void* dst, int dst_device, const void* src, int src_device, size_t data_size) {
    cudaError_t err = cudaMemcpyPeer(dst, dst_device, src, src_device, data_size);

    if (err) {
        fprintf(stderr, "Failed to copy memory via peer (direction: %d -> %d)\n", src_device, dst_device);
        exit(1);
    }
}

void cudaFreeSafe(void* ptr) {
    cudaError_t err = cudaFree(ptr);

    if (err) {
        fprintf(stderr, "Failed to make device memory free\n");
        exit(1);
    }
}

void cudaFreeHostSafe(void* ptr) {
    cudaError_t err = cudaFreeHost(ptr);

    if (err) {
        fprintf(stderr, "Failed to make host memory free\n");
        exit(1);
    }
}

