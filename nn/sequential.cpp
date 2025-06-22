#include "sequential.h"

namespace SushiAI 
{
    Sequential::Sequential(const std::vector<std::shared_ptr<Layer>>& layers) : layers(layers) 
    {

    }

    void Sequential::add(const std::shared_ptr<Layer>& layer) 
    {
        layers.push_back(layer);
    }

    void Sequential::remove(size_t index) 
    {
        if (index < layers.size())
            layers.erase(layers.begin() + index);
        else
            throw std::out_of_range("Sequential::remove(): index out of range");
    }

    std::shared_ptr<Tensor> Sequential::forward(const std::shared_ptr<Tensor>& input, bool training)
    {
        auto out = input;

        for (auto& layer : layers)
            out = layer -> forward(out, training);

        return out;
    }

    std::vector<std::shared_ptr<Tensor>> Sequential::parameters() const
    {
        std::vector<std::shared_ptr<Tensor>> params;

        for (auto& layer : layers) 
        {
            auto p = layer -> parameters();
            params.insert(params.end(), p.begin(), p.end());
        }

        return params;
    }
}
