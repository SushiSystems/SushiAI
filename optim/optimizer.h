#pragma once
#include <vector>
#include <memory>
#include <unordered_map>
#include "tensor.h"

namespace SushiAI 
{
    class Optimizer 
    {
        public:
            virtual ~Optimizer() = default;
            virtual void zeroGradient(const std::vector<std::shared_ptr<Tensor>>& parameters) = 0;
            virtual void step(const std::vector<std::shared_ptr<Tensor>>& parameters) = 0;
    };

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
}