#pragma once
#include <vector>
#include <memory>
#include <initializer_list>
#include <iostream>
#include <cassert>
#include <functional>

namespace SushiAI
{
    class Tensor : public std::enable_shared_from_this<Tensor>
    {
        private:
            std::vector<int> strides;
            int totalSize;
            float* device_data = nullptr;

            void calculateStrides();
            int getFlatIndex(std::initializer_list<int> indices) const;

        public:
            std::vector<float> data;
            std::vector<int> shape;

            bool requires_grad = false;
            std::vector<float> grad;
            std::function<void()> grad_fn;
            std::vector<std::shared_ptr<Tensor>> parents;

            Tensor(const std::vector<int>& shape, float fill = 0.0f, bool req_grad = false);

            static std::shared_ptr<Tensor> Zeros(const std::vector<int>& shape, bool req_grad = false);
            static std::shared_ptr<Tensor> Ones(const std::vector<int>& shape, bool req_grad = false);

            float& at(std::initializer_list<int> indices);
            const float& at(std::initializer_list<int> indices) const;

            void reshape(const std::vector<int>& newShape);
            void print(const std::string& name = "") const;

            void allocateDevice();
            void copyToDevice();
            void copyToHost();
            void freeDevice();
            float* devicePtr() const;

            const std::vector<int>& getShape() const { return shape; }
            std::vector<float>& getData() { return data; }
            const std::vector<float>& getData() const { return data; }
            std::vector<float>& getGrad() { return grad; }
            const std::vector<float>& getGrad() const { return grad; }

            void setGradFn(std::function<void()> fn, std::vector<std::shared_ptr<Tensor>> prnts)
            {
            grad_fn = std::move(fn);
            parents = std::move(prnts);
            }

            void backward();
    };
}
