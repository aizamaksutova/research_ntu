#include <cuda_runtime.h>
#include <stdio.h>

__global__ void stress_pcie(int num_iterations, int transfer_size_bytes)
{
    char* src = new char[transfer_size_bytes];
    char* dst = new char[transfer_size_bytes];
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < num_iterations; i++)
    {
        cudaEventRecord(start);
        cudaMemcpy(dst, src, transfer_size_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(src, dst, transfer_size_bytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);
    }

    cudaEventSynchronize(stop);
    float elapsed_time_ms;
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    printf("Time taken for %d iterations of %d-byte transfers: %f ms\n", num_iterations, transfer_size_bytes, elapsed_time_ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete[] src;
    delete[] dst;
}

int main()
{
    int num_iterations = 1000;
    int transfer_size_bytes = 1024 * 1024; // 1 MB

    stress_pcie<<<1, 1>>>(num_iterations, transfer_size_bytes);
    cudaDeviceSynchronize();
}
