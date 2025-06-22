#pragma once
#include <memory>
#include "tensor.h"

namespace SushiAI 
{
    class Loss 
    {
        public:
            virtual ~Loss() = default;
            virtual std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input, const std::shared_ptr<Tensor>& target) = 0;
    };

    class MSELoss : public Loss 
    {
        public:
            std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input, const std::shared_ptr<Tensor>& target) override;
    };

    class CrossEntropyLoss : public Loss 
    {
        public:
            std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input, const std::shared_ptr<Tensor>& target) override;
    };
}