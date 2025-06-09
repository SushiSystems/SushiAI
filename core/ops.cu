#include "ops.h"
#include <cuda_runtime.h>
#include <cmath>
#include <memory>
#include <algorithm>
#include <cassert>

namespace SushiAI
{
    // ------ CUDA KERNEL TANIMLARI ------

    __global__ void addKernel(const float* a, const float* b, float* result, int size)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < size)
            result[i] = a[i] + b[i];
    }

    __global__ void reluKernel(const float* input, float* output, int size)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < size)
            output[i] = fmaxf(0.0f, input[i]);
    }

    __global__ void sigmoidKernel(const float* input, float* output, int size)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < size)
            output[i] = 1.0f / (1.0f + expf(-input[i]));
    }

    __global__ void tanhKernel(const float* input, float* output, int size)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < size)
            output[i] = tanhf(input[i]);
    }

    __global__ void softmaxExpShiftKernel(const float* input, float* output, float shift, int size)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < size)
            output[i] = expf(input[i] - shift);
    }

    __global__ void softmaxNormalizeKernel(float* data, float sumExp, int size)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < size)
            data[i] /= sumExp;
    }
    __global__ void matmulKernel(const float* A, const float* B, float* C, int M, int K, int N)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < M && col < N)
        {
            float value = 0;

            for (int i = 0; i < K; ++i)
                value += A[row * K + i] * B[i * N + col];

            C[row * N + col] = value;
        }
    }

    // ------- HELPER -------
    inline dim3 getGrid(int size, int threads = 256)
    {
        return dim3((size + threads - 1) / threads);
    }

    // -------- AUTOGRAD+POINTER CUDA OPS --------

    std::shared_ptr<Tensor> add_cuda(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b)
    {
        assert(a -> getShape() == b -> getShape());

        int size = static_cast<int>(a -> getData().size());

        bool req_grad = a -> requires_grad || b -> requires_grad;
        auto result = std::make_shared<Tensor>(a -> getShape(), 0.0f, req_grad);

        a -> copyToDevice();
        b -> copyToDevice();
        result -> allocateDevice();

        addKernel << <getGrid(size), 256 >> > (a -> devicePtr(), b -> devicePtr(), result -> devicePtr(), size);
        cudaDeviceSynchronize();

        result -> copyToHost();

        if (req_grad)
        {
            auto a_ptr = a -> shared_from_this();
            auto b_ptr = b -> shared_from_this();

            auto result_ptr = result;
            result -> setGradFn([a_ptr, b_ptr, result_ptr]() 
            {
                for (size_t i = 0; i < result_ptr -> grad.size(); ++i)
                {
                    if (a_ptr -> requires_grad)
                        a_ptr -> grad[i] += result_ptr -> grad[i];
                    if (b_ptr -> requires_grad)
                        b_ptr -> grad[i] += result_ptr -> grad[i];
                }
            }, { a_ptr, b_ptr });
        }

        return result;
    }

    std::shared_ptr<Tensor> relu_cuda(const std::shared_ptr<Tensor>& t)
    {
        int size = static_cast<int>(t -> getData().size());
        auto result = std::make_shared<Tensor>(t -> getShape(), 0.0f, t -> requires_grad);

        t -> copyToDevice();
        result -> allocateDevice();

        reluKernel << <getGrid(size), 256 >> > (t -> devicePtr(), result -> devicePtr(), size);
        cudaDeviceSynchronize();

        result -> copyToHost();

        if (t -> requires_grad)
        {
            auto t_ptr = t -> shared_from_this();

            auto result_ptr = result;
            result -> setGradFn([t_ptr, result_ptr]() 
            {
                for (size_t i = 0; i < result_ptr -> grad.size(); ++i)
                    t_ptr->grad[i] += (result_ptr -> data[i] > 0 ? 1.0f : 0.0f) * result_ptr -> grad[i];
            }, { t_ptr });
        }

        return result;
    }

    std::shared_ptr<Tensor> sigmoid_cuda(const std::shared_ptr<Tensor>& t)
    {
        int size = static_cast<int>(t -> getData().size());
        auto result = std::make_shared<Tensor>(t -> getShape(), 0.0f, t -> requires_grad);

        t -> copyToDevice();
        result -> allocateDevice();

        sigmoidKernel << <getGrid(size), 256 >> > (t -> devicePtr(), result -> devicePtr(), size);
        cudaDeviceSynchronize();

        result -> copyToHost();

        if (t -> requires_grad)
        {
            auto t_ptr = t -> shared_from_this();

            auto result_ptr = result;
            result -> setGradFn([t_ptr, result_ptr]() 
            {
                for (size_t i = 0; i < result_ptr -> grad.size(); ++i)
                {
                    float sig = result_ptr -> data[i];
                    t_ptr -> grad[i] += sig * (1 - sig) * result_ptr -> grad[i];
                }
            }, { t_ptr });
        }

        return result;
    }

    std::shared_ptr<Tensor> tanh_cuda(const std::shared_ptr<Tensor>& t)
    {
        int size = static_cast<int>(t -> getData().size());
        auto result = std::make_shared<Tensor>(t -> getShape(), 0.0f, t -> requires_grad);

        t -> copyToDevice();
        result -> allocateDevice();

        tanhKernel << <getGrid(size), 256 >> > (t -> devicePtr(), result -> devicePtr(), size);
        cudaDeviceSynchronize();

        result->copyToHost();

        if (t->requires_grad)
        {
            auto t_ptr = t -> shared_from_this();
            auto result_ptr = result;
            result -> setGradFn([t_ptr, result_ptr]() 
            {
                for (size_t i = 0; i < result_ptr -> grad.size(); ++i)
                {
                    float tanhval = result_ptr -> data[i];
                    t_ptr -> grad[i] += (1.0f - tanhval * tanhval) * result_ptr -> grad[i];
                }
            }, { t_ptr });
        }

        return result;
    }

    // Softmax için hem GPU'da max bulmak hem normalize etmek için iki kernel, CPU yardýmý kullanýyoruz
    std::shared_ptr<Tensor> softmax_cuda(const std::shared_ptr<Tensor>& t)
    {
        int size = static_cast<int>(t -> getData().size());
        auto result = std::make_shared<Tensor>(t->getShape(), 0.0f, t -> requires_grad);

        float maxVal = *std::max_element(t -> getData().begin(), t -> getData().end());
        // CPU'da max bulmak þimdilik daha kolay

        t -> copyToDevice();
        result -> allocateDevice();

        // exp(x - max)
        softmaxExpShiftKernel << <getGrid(size), 256 >> > (t -> devicePtr(), result -> devicePtr(), maxVal, size);
        cudaDeviceSynchronize();

        result -> copyToHost();

        float sumExp = 0.0f;
        for (auto val : result -> getData())
            sumExp += val;

        result -> copyToDevice();
        softmaxNormalizeKernel << <getGrid(size), 256 >> > (result->devicePtr(), sumExp, size);
        cudaDeviceSynchronize();

        result -> copyToHost();

        // backward: sadece softmax'ýn kendi gradyaný
        if (t -> requires_grad)
        {
            auto t_ptr = t -> shared_from_this();

            auto result_ptr = result;
            result->setGradFn([t_ptr, result_ptr]() 
            {
                for (size_t i = 0; i < result_ptr -> grad.size(); ++i)
                {
                    float s = result_ptr -> data[i];
                    t_ptr -> grad[i] += s * (1 - s) * result_ptr -> grad[i]; // simplification
                }
            }, { t_ptr });
        }

        return result;
    }

    std::shared_ptr<Tensor> matmul_cuda(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b)
    {
        assert(a -> getShape().size() == 2 && b -> getShape().size() == 2);
        assert(a -> getShape()[1] == b -> getShape()[0]);

        int M = a -> getShape()[0];
        int K = a -> getShape()[1];
        int N = b -> getShape()[1];

        bool req_grad = a -> requires_grad || b -> requires_grad;
        auto result = std::make_shared<Tensor>(std::vector<int>{M, N}, 0.0f, req_grad);

        // GPU'ya kopya
        a -> copyToDevice();
        b -> copyToDevice();
        result -> allocateDevice();

        dim3 threads(16, 16);
        dim3 blocks((N + 15) / 16, (M + 15) / 16);

        matmulKernel << <blocks, threads >> > (a->devicePtr(), b->devicePtr(), result->devicePtr(), M, K, N);
        cudaDeviceSynchronize();

        result -> copyToHost();

        if (req_grad)
        {
            auto a_ptr = a -> shared_from_this();
            auto b_ptr = b -> shared_from_this();

            auto result_ptr = result;
            result -> setGradFn([a_ptr, b_ptr, result_ptr, M, K, N]() 
            {
                // CPU-side backward: brute-force, aynen ops.cpp ile ayný
                for (int i = 0; i < M; ++i)
                {
                    for (int l = 0; l < K; ++l)
                    {
                        float gradA = 0.0f;

                        for (int j = 0; j < N; ++j)
                            gradA += b_ptr->at({ l, j }) * result_ptr->grad[i * N + j];

                        a_ptr -> grad[i * K + l] += gradA;
                    }
                }
                for (int l = 0; l < K; ++l)
                {
                    for (int j = 0; j < N; ++j)
                    {
                        float gradB = 0.0f;

                        for (int i = 0; i < M; ++i)
                            gradB += a_ptr -> at({ i, l }) * result_ptr -> grad[i * N + j];

                        b_ptr -> grad[l * N + j] += gradB;
                    }
                }
            }, { a_ptr, b_ptr });
        }

        return result;
    }
}
