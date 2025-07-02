/**************************************************************************/
/*  tensor.h                                                              */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2025-present  Mustafa Garip & Sushi Systems              */
/*                                                                   	  */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

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
    #pragma region Tensor Class

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

            #pragma region Tensor Constructor

            /// Default Tensor constructor.
            Tensor(const std::vector<int>& shape, float fill = 0.0f, bool requiresGrad = false);

            #pragma endregion

            #pragma region Tensor Functions

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

            /// Clears the list of parent tensors.
            void clearParents() { parents.clear(); }

            #pragma region Backpropagation Mechanism, REFACTORING NEEDED

            /// Performs backpropagation starting from a scalar output tensor.
            void backward(bool retainGraph = false, bool clearExisting = true);

            /// Performs backpropagation using a custom gradient seed vector.
            void backward(const std::vector<float>& seed, bool retainGraph = false, bool clearExisting = true);

            #pragma endregion

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

    #pragma endregion
}
