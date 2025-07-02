/**************************************************************************/
/*  optimizer.cpp                                                         */
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

#include <cmath>
#include <algorithm>
#include "optimizer.h"

namespace SushiAI 
{
    #pragma region SDG

    SGD::SGD(float learningRate, float momentum, float weightDecay) : learningRate(learningRate), momentum(momentum), weightDecay(weightDecay) {}

    void SGD::zeroGradient(const std::vector<std::shared_ptr<Tensor>>& parameters)
    {
        for (auto& p : parameters)
            std::fill(p -> getGradient().begin(), p -> getGradient().end(), 0.0f);
    }

    void SGD::step(const std::vector<std::shared_ptr<Tensor>>& parameters)
    {
        for (auto& p : parameters)
        {
            auto& data = p -> getData();
            auto& grad = p -> getGradient();
            Tensor* key = p.get();

            auto& v = velocity[key];

            if (v.empty())
                v.assign(data.size(), 0.0f);

            for (size_t i = 0; i < data.size(); ++i)
            {
                float g = grad[i] + weightDecay * data[i];
                v[i] = momentum * v[i] + learningRate * g;
                data[i] -= v[i];
            }
        }
    }

    #pragma endregion

    #pragma region Adam

    Adam::Adam(float learningRate, float b1, float b2, float eps) : learningRate(learningRate), beta1(b1), beta2(b2), eps(eps), timeStep(0) {}

    void Adam::zeroGradient(const std::vector<std::shared_ptr<Tensor>>& parameters)
    {
        for (auto& p : parameters)
            std::fill(p -> getGradient().begin(), p -> getGradient().end(), 0.0f);
    }

    void Adam::step(const std::vector<std::shared_ptr<Tensor>>& parameters)
    {
        ++timeStep;
        float biasCorrection1 = 1.0f - (float)std::pow(beta1, timeStep);
        float biasCorrection2 = 1.0f - (float)std::pow(beta2, timeStep);

        for (auto& p : parameters)
        {
            auto& data = p -> getData();
            auto& grad = p -> getGradient();
            Tensor* key = p.get();

            auto& mt = meanMoment[key];
            auto& vt = varianceMoment[key];

            if (mt.empty())
                mt.assign(data.size(), 0.0f);
            if (vt.empty())
                vt.assign(data.size(), 0.0f);

            for (size_t i = 0; i < data.size(); ++i)
            {
                mt[i] = beta1 * mt[i] + (1.0f - beta1) * grad[i];
                vt[i] = beta2 * vt[i] + (1.0f - beta2) * grad[i] * grad[i];

                float mHat = mt[i] / biasCorrection1;
                float vHat = vt[i] / biasCorrection2;

                data[i] -= learningRate * mHat / (std::sqrt(vHat) + eps);
            }
        }
    }

    #pragma endregion
}