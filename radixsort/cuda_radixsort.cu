#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "../common/util.cu.h"
#include "../common/data.h"
#include <algorithm>
#define PAD "%-42s "

void testRadixSortCUB(int32_t* input, size_t input_size, int32_t* expected, size_t expected_size) {
    using I = uint64_t;
    const I size = input_size;
    const I ARRAY_BYTES = size * sizeof(int32_t);
    const I WARMUP_RUNS = 200;
    const I RUNS = 100;
    
    std::vector<int32_t> h_in(size);
    std::vector<int32_t> h_out(size, 0);

    for (I i = 0; i < size; ++i) {
        h_in[i] = input[i];
    }
    
    int32_t *d_in, *d_out;
    gpuAssert(cudaMalloc((void**)&d_in, ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_out, ARRAY_BYTES));
    
    gpuAssert(cudaMemcpy(d_in, h_in.data(), ARRAY_BYTES, cudaMemcpyHostToDevice));
    
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in, d_out, size);

    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    float * temp = (float *) malloc(sizeof(float) * RUNS);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (I i = 0; i < WARMUP_RUNS; ++i) {
        cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in, d_out, size);
        cudaDeviceSynchronize();
        gpuAssert(cudaPeekAtLastError());
    }

    for (I i = 0; i < RUNS; ++i) {
        cudaEventRecord(start, 0);
        cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in, d_out, size);
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(temp + i, start, stop);
        gpuAssert(cudaPeekAtLastError());
    }

    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in, d_out, size);
    cudaDeviceSynchronize();
    gpuAssert(cudaMemcpy(h_out.data(), d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost));
    
    bool test_passes = true;
    if (expected_size != size) {
        std::cout << "Radix Sort (CUB) Test Failed: Expected size=" << expected_size << " but got size=" << size << ".\n";
        test_passes = false;
    } else {
        for (I i = 0; i < size; ++i) {
            if (h_out[i] != expected[i]) {
                std::cout << "Radix Sort (CUB) Test Failed: Mismatch at index=" << i 
                          << " expected=" << expected[i] << " got=" << h_out[i] << ".\n";
                test_passes = false;
                break;
            }
        } 
    }

    if (test_passes) {
        compute_descriptors(temp, RUNS, 2 * ARRAY_BYTES);
    }

    free(temp);
    gpuAssert(cudaFree(d_in));
    gpuAssert(cudaFree(d_out));
    gpuAssert(cudaFree(d_temp_storage));
}

int main(int32_t argc, char *argv[]) {
    assert(argc == 3);
    size_t input_size;
    int32_t* input = read_i32_array(argv[1], &input_size);
    size_t expected_size;
    int32_t* expected = read_i32_array(argv[2], &expected_size);
    
    printf("%s:\n", argv[1]);
    printf(PAD, "Radix Sort (CUB):");
    testRadixSortCUB(input, input_size, expected, expected_size);
    
    free(input);
    free(expected);

    gpuAssert(cudaPeekAtLastError());
    return 0;
}
