#include "optimizer.h"
#include <cmath>
#include <algorithm>

namespace SushiAI 
{
    // ----- SGD -----
    SGD::SGD(float learningRate, float momentum, float weightDecay) : learningRate(learningRate), momentum(momentum), weightDecay(weightDecay) 
    {

    }

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

    // ----- Adam -----
    Adam::Adam(float learningRate, float b1, float b2, float eps) : learningRate(learningRate), beta1(b1), beta2(b2), eps(eps), timeStep(0) 
    {

    }

    void Adam::zeroGradient(const std::vector<std::shared_ptr<Tensor>>& params) 
    {
        for (auto& p : params)
            std::fill(p -> getGradient().begin(), p -> getGradient().end(), 0.0f);
    }

    void Adam::step(const std::vector<std::shared_ptr<Tensor>>& params) 
    {
        ++timeStep;
        float biasCorrection1 = 1.0f - (float)std::pow(beta1, timeStep);
        float biasCorrection2 = 1.0f - (float)std::pow(beta2, timeStep);

        for (auto& p : params) 
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
}