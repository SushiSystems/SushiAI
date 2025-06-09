#include "ops.h"
#include <cmath>
#include <memory>
#include <algorithm>
#include <cassert>

namespace SushiAI
{
    // Element-wise addition
    std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b)
    {
        assert(a -> getShape() == b -> getShape());

        bool req_grad = a -> requires_grad || b -> requires_grad;
        auto result = std::make_shared<Tensor>(a -> getShape(), 0.0f, req_grad);

        const auto& dataA = a -> getData();
        const auto& dataB = b -> getData();
        auto& resData = result -> getData();

        for (size_t i = 0; i < dataA.size(); ++i)
            resData[i] = dataA[i] + dataB[i];

        if (req_grad)
        {
            auto a_ptr = a -> shared_from_this();
            auto b_ptr = b -> shared_from_this();

            auto result_ptr = result;
            result -> setGradFn([a_ptr, b_ptr, result_ptr]() 
            {
                for (size_t i = 0; i < result_ptr -> grad.size(); ++i)
                {
                    if (a_ptr -> requires_grad)
                        a_ptr -> grad[i] += result_ptr -> grad[i];
                    if (b_ptr -> requires_grad)
                        b_ptr -> grad[i] += result_ptr -> grad[i];
                }
            }, { a_ptr, b_ptr });
        }

        return result;
    }

    // Element-wise ReLU
    std::shared_ptr<Tensor> relu(const std::shared_ptr<Tensor>& t)
    {
        auto result = std::make_shared<Tensor>(t->getShape(), 0.0f, t -> requires_grad);

        const auto& data = t -> getData();
        auto& resData = result -> getData();

        for (size_t i = 0; i < data.size(); ++i)
            resData[i] = std::max(0.0f, data[i]);

        if (t -> requires_grad)
        {
            auto t_ptr = t -> shared_from_this();
            auto result_ptr = result;
            result->setGradFn([t_ptr, result_ptr]() 
            {
                for (size_t i = 0; i < result_ptr -> grad.size(); ++i)
                    t_ptr -> grad[i] += (result_ptr -> data[i] > 0 ? 1.0f : 0.0f) * result_ptr -> grad[i];
            }, { t_ptr });
        }

        return result;
    }

    // Element-wise Sigmoid
    std::shared_ptr<Tensor> sigmoid(const std::shared_ptr<Tensor>& t)
    {
        auto result = std::make_shared<Tensor>(t->getShape(), 0.0f, t -> requires_grad);

        const auto& data = t -> getData();
        auto& resData = result -> getData();

        for (size_t i = 0; i < data.size(); ++i)
            resData[i] = 1.0f / (1.0f + std::exp(-data[i]));

        if (t -> requires_grad)
        {
            auto t_ptr = t -> shared_from_this();

            auto result_ptr = result;
            result -> setGradFn([t_ptr, result_ptr]() 
            {
                for (size_t i = 0; i < result_ptr -> grad.size(); ++i)
                {
                    float sig = result_ptr -> data[i];
                    t_ptr -> grad[i] += sig * (1 - sig) * result_ptr -> grad[i];
                }
            }, { t_ptr });
        }

        return result;
    }

    // Element-wise Tanh
    std::shared_ptr<Tensor> tanh(const std::shared_ptr<Tensor>& t)
    {
        auto result = std::make_shared<Tensor>(t->getShape(), 0.0f, t->requires_grad);
        const auto& data = t -> getData();
        auto& resData = result -> getData();

        for (size_t i = 0; i < data.size(); ++i)
            resData[i] = std::tanh(data[i]);

        if (t -> requires_grad)
        {
            auto t_ptr = t -> shared_from_this();

            auto result_ptr = result;
            result -> setGradFn([t_ptr, result_ptr]() 
            {
                for (size_t i = 0; i < result_ptr -> grad.size(); ++i)
                {
                    float tanhval = result_ptr -> data[i];
                    t_ptr -> grad[i] += (1.0f - tanhval * tanhval) * result_ptr -> grad[i];
                }
            }, { t_ptr });
        }

        return result;
    }

    // Softmax (vector only, cross-entropy compatible)
    std::shared_ptr<Tensor> softmax(const std::shared_ptr<Tensor>& t)
    {
        auto result = std::make_shared<Tensor>(t -> getShape(), 0.0f, t -> requires_grad);

        const auto& data = t -> getData();
        auto& resData = result->getData();

        float maxVal = *std::max_element(data.begin(), data.end());
        float sumExp = 0.0f;

        for (auto val : data)
            sumExp += std::exp(val - maxVal);
        for (size_t i = 0; i < data.size(); ++i)
            resData[i] = std::exp(data[i] - maxVal) / sumExp;

        // Softmax'ýn backward'ý tipik olarak cross-entropy ile birlikte kullanýlýr.
        // Burada sadece softmax'ýn kendi gradyaný örneði:
        if (t -> requires_grad)
        {
            auto t_ptr = t->shared_from_this();

            auto result_ptr = result;
            result -> setGradFn([t_ptr, result_ptr]() 
            {
                for (size_t i = 0; i < result_ptr -> grad.size(); ++i)
                {
                    float s = result_ptr -> data[i];
                    t_ptr -> grad[i] += s * (1 - s) * result_ptr -> grad[i]; // simplification!
                }
            }, { t_ptr });
        }

        return result;
    }

    // Matrix multiplication (2D only)
    std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b)
    {
        assert(a -> getShape().size() == 2 && b -> getShape().size() == 2);
        assert(a -> getShape()[1] == b -> getShape()[0]);

        int m = a -> getShape()[0];
        int k = a -> getShape()[1];
        int n = b -> getShape()[1];

        auto result = std::make_shared<Tensor>(std::vector<int>{ m, n }, 0.0f, a -> requires_grad || b -> requires_grad);

        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                float sum = 0.0f;

                for (int l = 0; l < k; ++l)
                    sum += a -> at({ i, l }) * b -> at({ l, j });

                result -> at({ i, j }) = sum;
            }
        }

        if (result -> requires_grad)
        {
            auto a_ptr = a -> shared_from_this();
            auto b_ptr = b -> shared_from_this();

            auto result_ptr = result;
            result -> setGradFn([a_ptr, b_ptr, result_ptr, m, k, n]() 
            {
                // dL/dA = dL/dR . B^T
                // dL/dB = A^T . dL/dR
                // Yalýn haliyle her elemana brute-force
                for (int i = 0; i < m; ++i)
                {
                    for (int l = 0; l < k; ++l)
                    {
                        float gradA = 0.0f;

                        for (int j = 0; j < n; ++j)
                            gradA += b_ptr -> at({ l, j }) * result_ptr -> grad[i * n + j];

                        a_ptr -> grad[i * k + l] += gradA;
                    }
                }
                for (int l = 0; l < k; ++l)
                {
                    for (int j = 0; j < n; ++j)
                    {
                        float gradB = 0.0f;

                        for (int i = 0; i < m; ++i)
                            gradB += a_ptr -> at({ i, l }) * result_ptr->grad[i * n + j];

                        b_ptr -> grad[l * n + j] += gradB;
                    }
                }
            }, { a_ptr, b_ptr });
        }

        return result;
    }

    // Argmax (CPU-side, gradyan yok)
    int argmax(const std::shared_ptr<Tensor>& t)
    {
        const auto& data = t -> getData();

        return std::distance(data.begin(), std::max_element(data.begin(), data.end()));
    }

    // Cross-entropy loss (for one-hot targets)
    float cross_entropy_loss(const std::shared_ptr<Tensor>& logits, const std::shared_ptr<Tensor>& targets)
    {
        assert(logits -> getShape() == targets -> getShape());

        float loss = 0.0f;

        const auto& logitData = logits -> getData();
        const auto& targetData = targets -> getData();

        for (size_t i = 0; i < logitData.size(); ++i)
            loss -= targetData[i] * std::log(logitData[i] + 1e-9f);

        return loss / logitData.size();
    }
}
