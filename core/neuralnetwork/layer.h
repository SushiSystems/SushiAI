/**************************************************************************/
/*  layer.h                                                               */
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
#include <memory>
#include <vector>
#include <string>
#include <random>
#include "initializer.h"
#include "tensor.h"
#include "ops.h"

namespace SushiAI 
{
    #pragma region Layer Class

    /// Abstract base class for all neural network layers.
    class Layer 
    {
        public:
            virtual ~Layer() = default;

            virtual std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input, bool training = true) = 0;
            std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) { return forward(input, true); }
            virtual std::string name() const = 0;
            virtual std::vector<std::shared_ptr<Tensor>> parameters() const { return {}; }

            virtual void resetState() {}
    };

    #pragma endregion

    #pragma region Linear (Dense) Layer

    /// Fully connected linear layer (a.k.a. Dense layer), output = input * weights^T + bias.
    class Linear : public Layer
    {
        public:
            Linear(int inFeatures, int outFeatures, std::shared_ptr<Initializer> weightInit, std::shared_ptr<Initializer> biasInit);

            std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input, bool training = true) override;
            std::string name() const override { return "Linear"; }
            std::vector<std::shared_ptr<Tensor>> parameters() const override { return { weights, bias }; }

            std::shared_ptr<Tensor> weights;
            std::shared_ptr<Initializer> weightInit;
            std::shared_ptr<Tensor> bias;
            std::shared_ptr<Initializer> biasInit;
    };

    #pragma endregion

    #pragma region Activation Layers

    /// Applies f(x) = max(0, x).
    class ReLU : public Layer
    {
        public:
            std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input, bool training = true) override
            {
                return relu(input);
            }

            std::string name() const override { return "ReLU"; }
    };

    /// Applies f(x) = max(alpha * x, x).
    class LeakyReLU : public Layer
    {
        public:
            float alpha;
            LeakyReLU(float alpha = 0.01f) : alpha(alpha) { }

            std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input, bool training = true) override
            {
                return leakyRelu(input, alpha);
            }

            std::string name() const override { return "Leaky ReLU"; }
    };

    /// Applies f(x) = 1 / (1 + exp(-x)).
    class Sigmoid : public Layer
    {
        public:
            std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input, bool training = true) override
            {
                return sigmoid(input);
            }

            std::string name() const override { return "Sigmoid"; }

    };

    /// Applies f(x) = tanh(x).
    class Tanh : public Layer
    {
        public:
            std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input, bool training = true) override
            {
                return tanh(input);
            }

            std::string name() const override { return "Tanh"; }
    };

    #pragma endregion

    #pragma region Regularization Layers

    /// Randomly zeroes out input values during training with probability p.
    class Dropout : public Layer 
    {
        private:
            float prob;

        public:
            Dropout(float p) : prob(p) {}

            std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input, bool training = true) override;
            std::string name() const override { return "Dropout (p = " + std::to_string(prob) + ")"; }
    };

    /// Normalizes input using running statistics and learnable scale/shift (for 2D input tensors).
    class BatchNorm : public Layer 
    {
        private:
            int numFeatures;
            float momentum, eps;

            std::shared_ptr<Tensor> gamma, beta;
            std::shared_ptr<Tensor> runningMean, runningVar;

        public:
            BatchNorm(int features, float momentum = 0.1f, float eps = 1e-5f) : numFeatures(features), momentum(momentum), eps(eps) 
            {
                gamma = Tensor::Ones({ features }, true);
                beta = Tensor::Zeros({ features }, true);
                runningMean = Tensor::Zeros({ features }, false);
                runningVar = Tensor::Ones({ features }, false);
            }

            std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input, bool training = true) override;
            std::string name() const override { return "BatchNorm (" + std::to_string(numFeatures) + ")"; }
            std::vector<std::shared_ptr<Tensor>> parameters() const override { return { gamma, beta }; }

            void resetState() 
            {
                std::fill(runningMean -> getData().begin(), runningMean -> getData().end(), 0.0f);
                std::fill(runningVar -> getData().begin(), runningVar -> getData().end(), 1.0f);
            }
    };

    #pragma endregion
} 
