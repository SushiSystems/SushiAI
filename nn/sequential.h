#pragma once
#include <vector>
#include <memory>
#include <string>
#include <iomanip> 
#include <numeric>
#include <iostream>
#include "layer.h"

namespace SushiAI 
{
    class Sequential : public Layer 
    {
        public:
            Sequential() = default;
            Sequential(const std::vector<std::shared_ptr<Layer>>& layers);

            std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input, bool training = true) override;
            std::string name() const override { return "Sequential"; }
            std::vector<std::shared_ptr<Tensor>> parameters() const override;

            void add(const std::shared_ptr<Layer>& layer);
            void remove(size_t index);
            
			size_t layersSize() const { return layers.size(); }
            std::shared_ptr<Layer> getLayer(size_t index) const
            {
                if (index < layers.size())
                    return layers[index];

                return nullptr;
			}

            void printSummary() const
            {
                std::cout << "=== Model Summary ===\n";

                size_t totalParams = 0;
                for (size_t i = 0; i < layers.size(); ++i)
                {
                    auto& layer = layers[i];
                    size_t layerParamCount = 0;
                    std::cout << "[" << i << "] " << layer->name() << ":\n";

                    auto params = layer->parameters();
                    for (size_t j = 0; j < params.size(); ++j)
                    {
                        auto& t = params[j];
                        size_t n = t->getTotalSize();
                        layerParamCount += n;

                        std::cout << "  Parameter #" << j
                            << " Shape: ";
                        for (int d : t->getShape())
                            std::cout << "[" << d << "]";
                        std::cout << " | Count: " << n;

                        // gradient sum
                        const auto& grad = t->getGradient();
                        float gradSum = std::accumulate(grad.begin(), grad.end(), 0.0f,
                            [](float acc, float g) { return acc + std::abs(g); });

                        std::cout << " | Gradient Sum: " << std::fixed << std::setprecision(6) << gradSum;

                        // first 3 gradient values
                        std::cout << " | Gradient[0..2]: ";
                        for (int k = 0; k < std::min(3, (int)grad.size()); ++k)
                            std::cout << std::fixed << std::setprecision(4)
                            << grad[k] << (k < 2 ? ", " : "");
                        std::cout << "\n";
                    }

                    totalParams += layerParamCount;
                    std::cout << "  -> Layer total params: " << layerParamCount << "\n";
                }

                std::cout << "=== Total trainable parameters: " << totalParams << " ===\n";
            }

        private:
            std::vector<std::shared_ptr<Layer>> layers;
    };
}
