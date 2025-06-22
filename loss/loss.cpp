#include <cmath>
#include <cassert>
#include "loss.h"
#include "ops.h"

namespace SushiAI
{
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
}