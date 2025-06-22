#include <random>
#include "layer.h"
#include "ops.h"

namespace SushiAI
{
    Linear::Linear(int in_features, int out_features, std::shared_ptr<Initializer> weightInitializer, std::shared_ptr<Initializer> biasInitializer) : weightInit(std::move(weightInitializer)), biasInit(std::move(biasInitializer))
    {
        weights = Tensor::Zeros({ in_features, out_features }, true);
        bias = Tensor::Zeros({ out_features }, true);

        // Enjeksiyonla gelen initializer’ı çağıralım
        weightInit -> initialize(weights);
        biasInit -> initialize(bias);
    }

    std::shared_ptr<Tensor> Linear::forward(const std::shared_ptr<Tensor>& input, bool training)
    {
        if (input -> getShape().size() == 1)
        {
            // Örn: [3] → [1,3] (tek örnek için batch size = 1)
            auto reshaped = std::make_shared<Tensor>(std::vector<int>{1, input -> getShape()[0]}, 0.0f, input -> requiresGradient);

            for (int i = 0; i < input -> getTotalSize(); ++i)
                reshaped -> at({ 0, i }) = input -> at({ i });

            reshaped->setGradientFunction([input, reshaped]() 
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
}