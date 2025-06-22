#pragma once
#include <vector>
#include <memory>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <functional>
#include <initializer_list>

namespace SushiAI
{
    /// @class Tensor
    /// Represents a multi-dimensional array with autograd support.
    class Tensor : public std::enable_shared_from_this<Tensor>
    {
        private:
            std::vector<int> strides;
            int totalSize;

            #pragma region Private Methods 

            /// Calculate strides based on shape.
			void calculateStrides(); 

            /// Converts tensor to flat array of floats.
			int getFlatIndex(std::initializer_list<int> indices) const;

            #pragma endregion

        public:
            std::vector<float> data;
            std::vector<int> shape;

            bool requiresGradient = false;
            std::vector<float> gradient;
            std::function<void()> gradientFunction;
            std::vector<std::shared_ptr<Tensor>> parents;

            #pragma region The Constructor and Factory Methods

            /// Default Tensor constructor.
            Tensor(const std::vector<int>& shape, float fill = 0.0f, bool requiresGrad = false);

            /// Tensor construction with zeros.
            static std::shared_ptr<Tensor> Zeros(const std::vector<int>& shape, bool requiresGrad = false);
            /// Tensor construction with ones.
            static std::shared_ptr<Tensor> Ones(const std::vector<int>& shape, bool requiresGrad = false);

            /// Accesses a tensor element using a flat index. (Read/Write)
            float& at(std::initializer_list<int> indices);
            /// Accesses a tensor element using a flat index. (Read)
            const float& at(std::initializer_list<int> indices) const;

            /// Reshape the tensor to a new shape, ensuring the total size remains the same.
            void reshape(const std::vector<int>& newShape);
            /// Prints the tensor’s shape, data, and gradients.
            void print(const std::string& name = "") const;
            
            #pragma endregion

            #pragma region Computation Graph

            /// Returns the list of tensors in topological order.
            std::vector<Tensor*> topologicalSort() const;

            /// Performs backpropagation starting from a scalar output tensor.
            void backward(bool retainGraph = false, bool clearExisting = true);
            /// Performs backpropagation using a custom gradient seed vector.
            void backward(const std::vector<float>& seed, bool retainGraph = false, bool clearExisting = true);

            /// Clears the list of parent tensors.
            void clearParents() { parents.clear(); }

            #pragma endregion

            #pragma region Set Gradient Function

            /// Sets the gradient function and its parent tensors for backpropagation.
            void setGradientFunction(std::function<void()> fn, std::vector<std::shared_ptr<Tensor>> prnts)
            {
                gradientFunction = std::move(fn);
                parents = std::move(prnts);
            }

            #pragma endregion

            #pragma region Getters

            int getTotalSize() const { return totalSize; };

            const std::vector<int>& getShape() const { return shape; }

            const std::vector<int>& getStrides() const { return strides; };

            const std::vector<std::shared_ptr<Tensor>>& getParents() const { return parents; };

            std::vector<float>& getData() { return data; }
            const std::vector<float>& getData() const { return data; }

            std::vector<float>& getGradient() { return gradient; }
            const std::vector<float>& getGradient() const { return gradient; }

            #pragma endregion
    };
}
