/**************************************************************************/
/*  optimizer.h                                                           */
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
#include <unordered_map>
#include "tensor.h"

namespace SushiAI 
{
    #pragma region Optimizer Class

    /// Abstract base class for all optimizers.
    class Optimizer 
    {
        public:
            virtual ~Optimizer() = default;
            virtual void zeroGradient(const std::vector<std::shared_ptr<Tensor>>& parameters) = 0;
            virtual void step(const std::vector<std::shared_ptr<Tensor>>& parameters) = 0;
    };

    #pragma endregion

    #pragma region Optimizers

    /// Stochastic Gradient Descent. Supports momentum and optional weight decay (L2 regularization).
    class SGD : public Optimizer 
    {
        public:
            SGD(float learningRate, float momentum = 0.0f, float weightDecay = 0.0f);
            void zeroGradient(const std::vector<std::shared_ptr<Tensor>>& parameters) override;
            void step(const std::vector<std::shared_ptr<Tensor>>& parameters) override;

            float getLearningRate() const { return learningRate; }
            float getMomentum() const { return momentum; }

        private:
            float learningRate;
            float momentum;
            float weightDecay;
            std::unordered_map<Tensor*, std::vector<float>> velocity;
    };

    /// Adaptive Moment Estimation. Combines momentum and RMSprop-like adaptive learning rates.
    class Adam : public Optimizer 
    {
        public:
            Adam(float learningRate, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f);
            void zeroGradient(const std::vector<std::shared_ptr<Tensor>>& parameters) override;
            void step(const std::vector<std::shared_ptr<Tensor>>& parameters) override;

            float getLearningRate() const { return learningRate; }
        private:
            float learningRate;
            float beta1;
            float beta2;
            float eps;
            int timeStep;
            std::unordered_map<Tensor*, std::vector<float>> meanMoment;
            std::unordered_map<Tensor*, std::vector<float>> varianceMoment;
    };

    #pragma endregion
}