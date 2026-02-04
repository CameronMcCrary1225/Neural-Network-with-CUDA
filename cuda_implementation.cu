#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cuda_implementation.cuh"
#include "base.h"
__global__ void hello_kernel() {
    printf("Hello from GPU thread %d\n", threadIdx.x);
}

__global__ void mat_add_cuda(const float* A, const float* B, float* out, u64 size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        out[idx] = A[idx] + B[idx];
    }
}
__global__ void mat_sub_cuda(const float* A, const float* B, float* out, u64 size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        out[idx] = A[idx] - B[idx];
    }
}
__global__ void mat_relu_cuda(float* out, const float* in, u64 size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        out[idx] = (in[idx] < 0) ? 0 : in[idx];
    }
}

__global__ void softmax_exp_kernel(const float* in, float* out, float* sum, u64 size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;

    float val = expf(in[idx]);
    out[idx] = val;
    atomicAdd(sum, val);
}
__global__ void mat_cross_entropy_kernel(const float* P, const float* Q, float* out, u64 size) {
    u64 idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        out[idx] = (P[idx] == 0.0f) ? 0.0f : P[idx] * -logf(Q[idx]);
    }
}
__global__ void softmax_normalize_kernel(float* out, float sum, u64 size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;

    out[idx] /= sum;
}
__global__ void mat_mul_kernel_nn(const float* A, const float* B, float* out,
    u32 A_rows, u32 A_cols, u32 B_cols)
{
    u32 row = blockIdx.y * blockDim.y + threadIdx.y;
    u32 col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < A_rows && col < B_cols)
    {
        float sum = 0.0f;
        for (u32 k = 0; k < A_cols; ++k)
        {
            sum += A[row * A_cols + k] * B[k * B_cols + col];
        }
        out[row * B_cols + col] = sum;
    }
}
void cuda_hello() {
    hello_kernel <<<1, 4 >>> ();
    cudaDeviceSynchronize();
}
extern "C" void mat_add_cuda(matrix* out, const matrix* a, const matrix* b) {
    if (!out || !a || !b) return;
    u64 size = (u64)out->rows * (u64)out->cols;

    // launch configuration
    const int blockSize = 256;
    int numBlocks = (int)((size + blockSize - 1) / blockSize);

    // NOTE: this assumes a->data, b->data, out->data point to CUDA-accessible memory
    // (cudaMalloc / cudaMallocManaged). If not, you must copy memory first.
    mat_add_cuda <<<numBlocks, blockSize >> > ((const f32*)a->data, (const f32*)b->data, (f32*)out->data, size);

    // check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("mat_add kernel launch failed: %s\n", cudaGetErrorString(err));
        return;
    }

    // wait for completion and check
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("mat_add cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        return;
    }
}
extern "C" void mat_sub_cuda(matrix* out, const matrix* a, const matrix* b) {
    if (!out || !a || !b) return;
    u64 size = (u64)out->rows * (u64)out->cols;

    // launch configuration
    const int blockSize = 256;
    int numBlocks = (int)((size + blockSize - 1) / blockSize);

    // NOTE: this assumes a->data, b->data, out->data point to CUDA-accessible memory
    // (cudaMalloc / cudaMallocManaged). If not, you must copy memory first.
    mat_sub_cuda <<<numBlocks, blockSize >>>((const f32*)a->data, (const f32*)b->data, (f32*)out->data, size);

    // check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("mat_sub kernel launch failed: %s\n", cudaGetErrorString(err));
        return;
    }

    // wait for completion and check
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("mat_sub cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        return;
    }
}
extern "C" void mat_relu_cuda(matrix* out, const matrix* in) {
    if (!out || !in) return;
    u64 size = (u64)out->rows * (u64)out->cols;

    // launch configuration
    const int blockSize = 256;
    int numBlocks = (int)((size + blockSize - 1) / blockSize);

    // NOTE: this assumes a->data, b->data, out->data point to CUDA-accessible memory
    // (cudaMalloc / cudaMallocManaged). If not, you must copy memory first.
    mat_relu_cuda <<<numBlocks, blockSize >>> ((f32*)out->data, (const f32*)in->data, size);

    // check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("matrelu kernel launch failed: %s\n", cudaGetErrorString(err));
        return;
    }

    // wait for completion and check
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("mat_relu cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        return;
    }
}
extern "C" void mat_softmax_cuda(matrix* out, const matrix* in) {
    if (!out || !in || out->rows != in->rows || out->cols != in->cols) return;

    u64 size = (u64)out->rows * out->cols;
    float* d_sum;
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemset(d_sum, 0, sizeof(float));

    int blockSize = 256;
    int numBlocks = (int)((size + blockSize - 1) / blockSize);

    softmax_exp_kernel << <numBlocks, blockSize >> > (in->data, out->data, d_sum, size);
    cudaDeviceSynchronize();

    float sum;
    cudaMemcpy(&sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_sum);

    softmax_normalize_kernel <<<numBlocks, blockSize >>> (out->data, sum, size);
    cudaDeviceSynchronize();
}
extern "C" void mat_cross_entropy_cuda(matrix* out, const matrix* p, const matrix* q) {
    if (!out || !p || !q) return;

    u64 size = (u64)out->rows * (u64)out->cols;

    const int blockSize = 256;
    int numBlocks = (int)((size + blockSize - 1) / blockSize);

    mat_cross_entropy_kernel << <numBlocks, blockSize >> > (
        (const float*)p->data,
        (const float*)q->data,
        (float*)out->data,
        size
        );

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("mat_cross_entropy kernel launch failed: %s\n", cudaGetErrorString(err));
        return;
    }

    // Wait for GPU to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("mat_cross_entropy cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        return;
    }
}
extern "C" void mat_mul_cuda(matrix* out, const matrix* a, const matrix* b)
{
    if (!out || !a || !b) return;

    u32 A_rows = a->rows;
    u32 A_cols = a->cols;
    u32 B_cols = b->cols;

    dim3 block(16, 16);
    dim3 grid((B_cols + block.x - 1) / block.x, (A_rows + block.y - 1) / block.y);

    mat_mul_kernel_nn << <grid, block >> > ((const float*)a->data, (const float*)b->data, (float*)out->data,
        A_rows, A_cols, B_cols);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("mat_mul kernel launch failed: %s\n", cudaGetErrorString(err));

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        printf("mat_mul cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
}