#include "tensor.h"
#include "cuda_utils.h"
#include <random>
#include <numeric>

namespace SushiAI
{
    #pragma region Tensor Functions

    Tensor::Tensor(const std::vector<int>& shape, float fill, bool req_grad) : shape(shape), requires_grad(req_grad), totalSize(1)
    {
        for (int s : shape)
            totalSize *= s;

        data.resize(totalSize, fill);
        grad.resize(totalSize, 0.0f);

        calculateStrides();
    }

    std::shared_ptr<Tensor> Tensor::Zeros(const std::vector<int>& shape, bool req_grad)
    {
        return std::make_shared<Tensor>(shape, 0.0f, req_grad);
    }

    std::shared_ptr<Tensor> Tensor::Ones(const std::vector<int>& shape, bool req_grad)
    {
        return std::make_shared<Tensor>(shape, 1.0f, req_grad);
    }

    void Tensor::calculateStrides()
    {
        strides.resize(shape.size());

        int stride = 1;
        for (size_t i = 0; i < shape.size(); ++i)
        {
            strides[i] = stride;
            stride *= shape[i];
        }
    }

    int Tensor::getFlatIndex(std::initializer_list<int> indices) const
    {
        assert(indices.size() == shape.size());

        int idx = 0, i = 0;
        for (int ind : indices)
        {
            assert(ind >= 0 && ind < shape[i]);
            idx += ind * strides[i];
            i++;
        }

        return idx;
    }

    float& Tensor::at(std::initializer_list<int> indices)
    {
        return data[getFlatIndex(indices)];
    }

    const float& Tensor::at(std::initializer_list<int> indices) const
    {
        return data[getFlatIndex(indices)];
    }

    void Tensor::reshape(const std::vector<int>& newShape)
    {
        int newSize = 1;

        for (int dim : newShape)
            newSize *= dim;

        assert(newSize == totalSize);

        shape = newShape;
        calculateStrides();
    }

    void Tensor::print(const std::string& name) const
    {
        if (!name.empty())
            std::cout << name << " ";

        std::cout << "Tensor shape: [";

        for (size_t i = 0; i < shape.size(); ++i)
        {
            std::cout << shape[i];
            if (i < shape.size() - 1) std::cout << ", ";
        }

        std::cout << "]\n[";

        for (size_t i = 0; i < data.size(); ++i)
        {
            std::cout << data[i];
            if (i < data.size() - 1) std::cout << ", ";
        }

        std::cout << "]\n";

        if (!grad.empty()) 
        {
            std::cout << "Grad: [";

            for (size_t i = 0; i < grad.size(); ++i) 
            {
                std::cout << grad[i];

                if (i < grad.size() - 1) 
                    std::cout << ", ";
            }

            std::cout << "]\n";
        }
    }

    void Tensor::backward()
    {
        grad.assign(grad.size(), 1.0f);

        std::vector<std::shared_ptr<Tensor>> stack = { shared_from_this() };

        while (!stack.empty())
        {
            auto node = stack.back();
            stack.pop_back();

            if (node -> grad_fn)
                node -> grad_fn();

            for (auto& p : node -> parents)
                if (p && !p -> grad_fn)
                    stack.push_back(p);
        }
    }


    #pragma endregion

    #pragma region CUDA

    void Tensor::allocateDevice()
    {
        if (!device_data)
            CUDA_CHECK(cudaMalloc(&device_data, totalSize * sizeof(float)));
    }

    void Tensor::copyToDevice()
    {
        if (!device_data)
            allocateDevice();

        CUDA_CHECK(cudaMemcpy(device_data, data.data(), totalSize * sizeof(float), cudaMemcpyHostToDevice));
    }

    void Tensor::copyToHost()
    {
        if (device_data)
            CUDA_CHECK(cudaMemcpy(data.data(), device_data, totalSize * sizeof(float), cudaMemcpyDeviceToHost));
    }

    void Tensor::freeDevice()
    {
        if (device_data)
        {
            CUDA_CHECK(cudaFree(device_data));
            device_data = nullptr;
        }
    }

    float* Tensor::devicePtr() const
    {
        return device_data;
    }

    #pragma endregion
}
