/**************************************************************************/
/*  loss.cpp                                                              */
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
#include <cassert>
#include "loss.h"
#include "ops.h"

namespace SushiAI
{
    #pragma region Loss Function Forwards

    std::shared_ptr<Tensor> MSELoss::forward(const std::shared_ptr<Tensor>& input, const std::shared_ptr<Tensor>& target)
    {
        const auto& inputData = input -> getData();
        const auto& targetData = target -> getData();
        size_t N = inputData.size();

        assert(N == targetData.size());

        float sum = 0.0f;
        for (size_t i = 0; i < N; ++i) 
        {
            float diff = inputData[i] - targetData[i];
            sum += diff * diff;
        }

        float lossValue = sum / static_cast<float>(N);
        auto loss = std::make_shared<Tensor>(std::vector<int>{1}, lossValue, true);

        loss -> setGradientFunction([input, target, loss, N]() 
        {
            float gradOut = loss -> getGradient()[0];
            auto& inGrad = input -> getGradient();
            const auto& inData = input -> getData();
            const auto& tData = target -> getData();

            for (size_t i = 0; i < N; ++i) 
                inGrad[i] += gradOut * 2.0f * (inData[i] - tData[i]) / static_cast<float>(N);

        }, { input });

        return loss;
    }

    std::shared_ptr<Tensor> CrossEntropyLoss::forward(const std::shared_ptr<Tensor>& input, const std::shared_ptr<Tensor>& target) 
    {
        return crossEntropyLoss(input, target);
    }

    #pragma endregion
}