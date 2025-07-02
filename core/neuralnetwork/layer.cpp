/**************************************************************************/
/*  layer.cpp                                                             */
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

#include <random>
#include "layer.h"
#include "ops.h"

namespace SushiAI
{
    #pragma region Linear (Dense) Layer

    Linear::Linear(int inFeatures, int outFeatures, std::shared_ptr<Initializer> weightInitializer, std::shared_ptr<Initializer> biasInitializer) : weightInit(std::move(weightInitializer)), biasInit(std::move(biasInitializer))
    {
        weights = Tensor::Zeros({ inFeatures, outFeatures }, true);
        bias = Tensor::Zeros({ outFeatures }, true);

        weightInit -> initialize(weights);
        biasInit -> initialize(bias);
    }

    #pragma endregion

    #pragma region Forward Functions

    std::shared_ptr<Tensor> Linear::forward(const std::shared_ptr<Tensor>& input, bool training)
    {
        if (input -> getShape().size() == 1)
        {
            auto reshaped = std::make_shared<Tensor>(std::vector<int>{1, input -> getShape()[0]}, 0.0f, input -> requiresGradient);

            for (int i = 0; i < input -> getTotalSize(); ++i)
                reshaped -> at({ 0, i }) = input -> at({ i });

            reshaped -> setGradientFunction([input, reshaped]() 
            {
                for (int i = 0; i < input -> getTotalSize(); ++i)
                    input -> gradient[i] += reshaped -> gradient[i];
            }, { input });

            return this -> forward(reshaped, training);
        }

        assert(input -> getShape().size() == 2);
        assert(input -> getShape()[1] == weights -> getShape()[0]);

        auto out = mul(input, weights);

        return add(out, bias);
    }

    std::shared_ptr<Tensor> Dropout::forward(const std::shared_ptr<Tensor>& input, bool training)
    {
        if (!training || prob <= 0.0f)
            return input;

        auto out = Tensor::Zeros(input -> getShape(), false);

        std::mt19937 gen(std::random_device{}());
        std::bernoulli_distribution dist(1.0f - prob);

        auto& inData = input -> getData();
        auto& outData = out -> getData();

        float scale = 1.0f / (1.0f - prob);
        for (size_t i = 0; i < inData.size(); ++i)
            outData[i] = dist(gen) ? inData[i] * scale : 0.0f;

        return out;
    }

    std::shared_ptr<Tensor> BatchNorm::forward(const std::shared_ptr<Tensor>& input, bool training)
    {
        auto shape = input -> getShape();
        assert(shape.size() == 2 && shape[1] == numFeatures);

        int batch = shape[0];
        auto out = Tensor::Zeros(shape, input -> requiresGradient);

        auto& inData = input -> getData();
        auto& outData = out -> getData();
        auto& muData = runningMean -> getData();
        auto& varData = runningVar -> getData();
        auto& gmData = gamma -> getData();
        auto& bData = beta -> getData();

        // compute batch stats
        std::vector<float> batchVar(numFeatures, 0.0f);
        std::vector<float> batchMean(numFeatures, 0.0f);

        for (int i = 0; i < batch; ++i)
        {
            for (int f = 0; f < numFeatures; ++f)
                batchMean[f] += inData[i * numFeatures + f];
        }

        for (auto& m : batchMean)
            m /= batch;

        for (int i = 0; i < batch; ++i)
        {
            for (int f = 0; f < numFeatures; ++f)
            {
                float d = inData[i * numFeatures + f] - batchMean[f];
                batchVar[f] += d * d;
            }
        }

        for (auto& v : batchVar)
            v = v / batch;

        if (training)
        {
            for (int f = 0; f < numFeatures; ++f)
            {
                muData[f] = momentum * batchMean[f] + (1 - momentum) * muData[f];
                varData[f] = momentum * batchVar[f] + (1 - momentum) * varData[f];
            }
        }

        // normalize
        const float* varPtr = training ? batchVar.data() : varData.data();
        const float* meanPtr = training ? batchMean.data() : muData.data();

        for (int i = 0; i < batch; ++i)
        {
            for (int f = 0; f < numFeatures; ++f)
            {
                int idx = i * numFeatures + f;
                outData[idx] = ((inData[idx] - meanPtr[f]) / std::sqrt(varPtr[f] + eps)) * gmData[f] + bData[f];
            }
        }
        return out;
    }

    #pragma endregion
}