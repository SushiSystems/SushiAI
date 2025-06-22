#pragma once
#include <memory>
#include <vector>
#include <string>
#include <random>
#include "initializer.h"
#include "tensor.h"
#include "ops.h"

namespace SushiAI 
{
    #pragma region Layer Class

    class Layer 
    {
        public:
            virtual ~Layer() = default;

            virtual std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input, bool training = true) = 0;
            std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) { return forward(input, true); }
            virtual std::string name() const = 0;
            virtual std::vector<std::shared_ptr<Tensor>> parameters() const { return {}; }

            virtual void resetState() {}
    };

    #pragma endregion

    // Linear (Dense) Layer
    class Linear : public Layer 
    {
        public:
            Linear(int in_features, int out_features, std::shared_ptr<Initializer> weightInit, std::shared_ptr<Initializer> biasInit);

            std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input, bool training = true) override;
            std::string name() const override { return "Linear"; }
            std::vector<std::shared_ptr<Tensor>> parameters() const override { return { weights, bias }; }

            std::shared_ptr<Tensor> weights;            
            std::shared_ptr<Initializer> weightInit;
            std::shared_ptr<Tensor> bias;
            std::shared_ptr<Initializer> biasInit;
    };

    class ReLU : public Layer
    {
        public:
            std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input, bool training = true) override
            {
                return relu(input);
            }

            std::string name() const override { return "ReLU"; }
    };

    class LeakyReLU : public Layer
    {
        public:
            float alpha;
            LeakyReLU(float alpha = 0.01f) : alpha(alpha) {}

            std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input, bool training = true) override
            {
                return leakyRelu(input, alpha);
            }

            std::string name() const override { return "Leaky ReLU"; }
    };

    class Sigmoid : public Layer
    {
        public:
            std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input, bool training = true) override
            {
                return sigmoid(input);
            }

            std::string name() const override { return "Sigmoid"; }

    };

    class Tanh : public Layer
    {
        public:
            std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input, bool training = true) override
            {
                return tanh(input);
            }

            std::string name() const override { return "Tanh"; }
    };

    #pragma region Regularization Layers

    // Dropout Layer
    class Dropout : public Layer 
    {
        private:
            float prob;

        public:
            Dropout(float p) : prob(p) {}

            std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input, bool training = true) override 
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

            std::string name() const override { return "Dropout(p=" + std::to_string(prob) + ")"; }
    };

    // Batch Normalization Layer (for 2D inputs: [batch, features])
    class BatchNorm : public Layer 
    {
        private:
            int numFeatures;
            float momentum, eps;

            std::shared_ptr<Tensor> gamma, beta;
            std::shared_ptr<Tensor> runningMean, runningVar;

        public:
            BatchNorm(int features, float momentum = 0.1f, float eps = 1e-5f) : numFeatures(features), momentum(momentum), eps(eps) 
            {
                gamma = Tensor::Ones({ features }, true);
                beta = Tensor::Zeros({ features }, true);
                runningMean = Tensor::Zeros({ features }, false);
                runningVar = Tensor::Ones({ features }, false);
            }

            std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input, bool training = true) override 
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

            std::string name() const override { return "BatchNorm(" + std::to_string(numFeatures) + ")"; }
            std::vector<std::shared_ptr<Tensor>> parameters() const override { return { gamma, beta }; }

            void resetState() 
            {
                std::fill(runningMean -> getData().begin(), runningMean -> getData().end(), 0.0f);
                std::fill(runningVar -> getData().begin(), runningVar -> getData().end(), 1.0f);
            }
    };

    #pragma endregion
} 
