/**************************************************************************/
/*  ops.cpp                                                               */
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

#include <cmath>
#include <vector>
#include <memory>
#include <cassert>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include "tensor.h"
#include "ops.h"

namespace SushiAI
{
    #pragma region Tensor Operations

    std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b)
    {
        auto sA = a -> getShape();
        auto sB = b -> getShape();
        int ndim = std::max((int)sA.size(), (int)sB.size());
        sA.insert(sA.begin(), ndim - (int)sA.size(), 1);
        sB.insert(sB.begin(), ndim - (int)sB.size(), 1);

        std::vector<int> sResult(ndim);
        for (int i = 0; i < ndim; ++i)
        {
            if (sA[i] == sB[i] || sA[i] == 1 || sB[i] == 1)
                sResult[i] = std::max(sA[i], sB[i]);
            else
                throw std::invalid_argument("add: shapes not broadcastable");
        }

        auto result = std::make_shared<Tensor>(sResult, 0.0f, a -> requiresGradient || b -> requiresGradient);

        auto origStA = a -> getStrides();
        auto origStB = b -> getStrides();

        std::vector<int> stA(ndim, 0), stB(ndim, 0);

        int shiftA = ndim - (int)origStA.size();
        int shiftB = ndim - (int)origStB.size();

        for (int i = 0; i < (int)origStA.size(); ++i)
            stA[i + shiftA] = origStA[i];
        for (int i = 0; i < (int)origStB.size(); ++i)
            stB[i + shiftB] = origStB[i];

        int N = result -> getTotalSize();
        auto& dA = a -> data;
        auto& dB = b -> data;
        auto& dR = result -> data;

        std::vector<int> idx(ndim);
        for (int flat = 0; flat < N; ++flat)
        {
            int tmp = flat, offA = 0, offB = 0;
            for (int d = ndim - 1; d >= 0; --d)
            {
                int dim = sResult[d];

                idx[d] = tmp % dim;
                tmp /= dim;

                int iA = (sA[d] == 1 ? 0 : idx[d]);
                int iB = (sB[d] == 1 ? 0 : idx[d]);

                offA += iA * stA[d];
                offB += iB * stB[d];
            }
            dR[flat] = dA[offA] + dB[offB];
        }

        if (result -> requiresGradient)
        {
            auto a_ptr = a;
            auto b_ptr = b;
            auto result_ptr = result;

            result -> setGradientFunction([a_ptr, b_ptr, result_ptr, sA, sB, stA, stB, sResult]()
            {
                const auto& gradR = result_ptr -> getGradient();
                auto& gradA = a_ptr -> getGradient();
                auto& gradB = b_ptr -> getGradient();
                int N = (int)gradR.size();
                std::vector<int> idx(sResult.size());

                for (int flat = 0; flat < N; ++flat)
                {
                    int tmp = flat, offA = 0, offB = 0;
                    for (int d = (int)sResult.size() - 1; d >= 0; --d)
                    {
                        int dim = sResult[d];

                        idx[d] = tmp % dim;
                        tmp /= dim;

                        int iA = (sA[d] == 1 ? 0 : idx[d]);
                        int iB = (sB[d] == 1 ? 0 : idx[d]);

                        offA += iA * stA[d];
                        offB += iB * stB[d];
                    }
                    if (a_ptr -> requiresGradient)
                        gradA[offA] += gradR[flat];
                    if (b_ptr -> requiresGradient)
                        gradB[offB] += gradR[flat];
                }
            }, { a_ptr, b_ptr });
        }

        return result;
    }

    std::shared_ptr<Tensor> mul(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b)
    {
        auto sA = a -> getShape();
        auto sB = b -> getShape();

        if (sA.size() == 2 && sB.size() == 2) 
        {
            if (sA[1] != sB[0])
                throw std::invalid_argument("mul: inner dimensions must match for 2D case");

            return matmul(a, b);
        }
        else if (sA.size() == 3 && sB.size() == 3) 
        {
            int batch = sA[0];
            int M = sA[1];
            int K = sA[2];
            int K2 = sB[1];
            int N = sB[2];

            if (batch != sB[0] || K != K2)
                throw std::invalid_argument("mul: batch size or inner dim mismatch for 3D case");

            auto result = std::make_shared<Tensor>(std::vector<int>{batch, M, N}, 0.0f, a -> requiresGradient || b -> requiresGradient);

            for (int bi = 0; bi < batch; ++bi) 
            {
                int offA = bi * (M * K);
                int offB = bi * (K * N);
                int offR = bi * (M * N);

                for (int i = 0; i < M; ++i) 
                {
                    for (int kk = 0; kk < K; ++kk) 
                    {
                        float a_val = a -> data[offA + i * K + kk];

                        for (int j = 0; j < N; ++j) 
                        {
                            result -> data[offR + i * N + j] +=
                                a_val * b -> data[offB + kk * N + j];
                        }
                    }
                }
            }

            if (result -> requiresGradient) 
            {
                auto a_ptr = a;
                auto b_ptr = b;
                auto result_ptr = result;

                result -> setGradientFunction([a_ptr, b_ptr, result_ptr, batch, M, K, N]() 
                {
                    const auto& gR = result_ptr -> gradient;
                    auto& gA = a_ptr -> gradient;
                    auto& gB = b_ptr -> gradient;

                    for (int bi = 0; bi < batch; ++bi) 
                    {
                        int offA = bi * (M * K);
                        int offB = bi * (K * N);
                        int offR = bi * (M * N);

                        for (int i = 0; i < M; ++i) 
                        {
                            for (int kk = 0; kk < K; ++kk) 
                            {
                                float sumA = 0.0f;

                                for (int j = 0; j < N; ++j) 
                                {
                                    float grad_out = gR[offR + i * N + j];

                                    gB[offB + kk * N + j] += a_ptr -> data[offA + i * K + kk] * grad_out;
                                    sumA += grad_out * b_ptr -> data[offB + kk * N + j];
                                }

                                gA[offA + i * K + kk] += sumA;
                            }
                        }
                    }
                }, { a_ptr, b_ptr });
            }

            return result;
        }
        else
            throw std::invalid_argument("mul: unsupported tensor ranks");
    }

    std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b)
    {
        const auto& A = a -> getData();
        const auto& B = b -> getData();
        int m = a -> getShape()[0];
        int k = a -> getShape()[1];
        int n = b -> getShape()[1];

        auto result = std::make_shared<Tensor>(std::vector<int>{ m, n }, 0.0f, a -> requiresGradient || b -> requiresGradient);
        auto& R = result -> getData();

        for (int i = 0; i < m; ++i)
        {
            int rowA = i * k;
            int rowR = i * n;

            for (int l = 0; l < k; ++l)
            {
                float a_val = A[rowA + l];
                int rowB = l * n;

                for (int j = 0; j < n; ++j)
                    R[rowR + j] += a_val * B[rowB + j];
            }
        }

        if (result -> requiresGradient)
        {
            auto a_ptr = a;
            auto b_ptr = b;
            auto result_ptr = result;

            result -> setGradientFunction([a_ptr, b_ptr, result_ptr, m, k, n]()
            {
                const auto& A = a_ptr -> getData();
                const auto& B = b_ptr -> getData();

                const auto& dR = result_ptr -> gradient;
                auto& dA = a_ptr -> gradient;
                auto& dB = b_ptr -> gradient;

                for (int i = 0; i < m; ++i)
                {
                    int rowA = i * k;
                    int rowR = i * n;

                    for (int l = 0; l < k; ++l)
                    {
                        float sum = 0.0f;
                        int rowB = l * n;

                        for (int j = 0; j < n; ++j)
                            sum += dR[rowR + j] * B[rowB + j];

                        dA[rowA + l] += sum;
                    }
                }

                for (int l = 0; l < k; ++l)
                {
                    int rowB = l * n;

                    for (int j = 0; j < n; ++j)
                    {
                        float sum = 0.0f;

                        for (int i = 0; i < m; ++i)
                            sum += A[i * k + l] * dR[i * n + j];

                        dB[rowB + j] += sum;
                    }
                }
            }, { a_ptr, b_ptr });
        }

        return result;
    }

    std::shared_ptr<Tensor> slice(const std::shared_ptr<Tensor>& t, int batchIdx)
    {
        const auto& shape = t -> getShape();
        int D = shape.size();
        assert((D == 2 || D == 3) && "slice: only 2D or 3D tensors supported");

        int B = shape[0];
        assert(batchIdx >= 0 && batchIdx < B && "slice: index out of range");

        std::vector<int> subShape(shape.begin() + 1, shape.end());  // [M] veya [M,N]
        int subSize = 1;
        for (int d : subShape) subSize *= d;

        auto view = std::make_shared<Tensor>(subShape, 0.0f, t -> requiresGradient);
        auto& dst = view -> getData();
        const auto& src = t -> getData();

        int offset = batchIdx * subSize;
        std::copy(src.begin() + offset, src.begin() + offset + subSize, dst.begin());

        if (t -> requiresGradient)
        {
            auto t_ptr = t;
            auto view_ptr = view;
            view -> setGradientFunction([t_ptr, view_ptr, offset, subSize]()
            {
                const auto& gV = view_ptr -> getGradient();
                auto& gT = t_ptr -> getGradient();

                for (int i = 0; i < subSize; ++i)
                    gT[offset + i] += gV[i];
            }, { t_ptr });
        }

        return view;
    }

    #pragma endregion

    #pragma region Activation Functions

    std::shared_ptr<Tensor> relu(const std::shared_ptr<Tensor>& t)
    {
        auto result = std::make_shared<Tensor>(t -> getShape(), 0.0f, t -> requiresGradient);

        const auto& data = t -> getData();
        auto& resultData = result -> getData();

        for (size_t i = 0; i < data.size(); ++i)
            resultData[i] = std::max(0.0f, data[i]);

        if (t -> requiresGradient)
        {
            auto t_ptr = t;
            auto result_ptr = result;
            result -> setGradientFunction([t_ptr, result_ptr]()
            {
                for (size_t i = 0; i < result_ptr -> gradient.size(); ++i)
                    t_ptr -> gradient[i] += (result_ptr -> data[i] > 0 ? 1.0f : 0.0f) * result_ptr -> gradient[i];
            }, { t_ptr });
        }

        return result;
    }

    std::shared_ptr<Tensor> leakyRelu(const std::shared_ptr<Tensor>& t, float alpha)
    {
        auto result = std::make_shared<Tensor>(t -> getShape(), 0.0f, t -> requiresGradient);

        const auto& data = t -> getData();
        auto& resultData = result -> getData();

        for (size_t i = 0; i < data.size(); ++i)
            resultData[i] = (data[i] > 0.0f ? data[i] : alpha * data[i]);

        if (t -> requiresGradient)
        {
            auto t_ptr = t -> shared_from_this();
            auto result_ptr = result;

            result -> setGradientFunction([t_ptr, result_ptr, alpha]()
            {
                const auto& outGrad = result_ptr -> getGradient();
                auto& inGrad = t_ptr -> getGradient();
                const auto& x = t_ptr -> getData();

                for (size_t i = 0; i < x.size(); ++i)
                {
                    float grad_coeff = (x[i] > 0.0f ? 1.0f : alpha);
                    inGrad[i] += grad_coeff * outGrad[i];
                }
            }, { t_ptr });
        }

        return result;
    }

    std::shared_ptr<Tensor> sigmoid(const std::shared_ptr<Tensor>& t)
    {
        auto result = std::make_shared<Tensor>(t -> getShape(), 0.0f, t -> requiresGradient);

        const auto& data = t -> getData();
        auto& resultData = result -> getData();

        for (size_t i = 0; i < data.size(); ++i)
            resultData[i] = 1.0f / (1.0f + std::exp(-data[i]));

        if (t -> requiresGradient)
        {
            auto t_ptr = t -> shared_from_this();

            auto result_ptr = result;
            result -> setGradientFunction([t_ptr, result_ptr]()
            {
                for (size_t i = 0; i < result_ptr -> gradient.size(); ++i)
                {
                    float sig = result_ptr -> data[i];
                    t_ptr -> gradient[i] += sig * (1 - sig) * result_ptr -> gradient[i];
                }
            }, { t_ptr });
        }

        return result;
    }

    std::shared_ptr<Tensor> tanh(const std::shared_ptr<Tensor>& t)
    {
        auto result = std::make_shared<Tensor>(t -> getShape(), 0.0f, t -> requiresGradient);
        const auto& data = t -> getData();
        auto& resultData = result -> getData();

        for (size_t i = 0; i < data.size(); ++i)
            resultData[i] = std::tanh(data[i]);

        if (t -> requiresGradient)
        {
            auto t_ptr = t -> shared_from_this();

            auto result_ptr = result;
            result -> setGradientFunction([t_ptr, result_ptr]()
            {
                for (size_t i = 0; i < result_ptr -> gradient.size(); ++i)
                {
                    float tanhval = result_ptr -> data[i];
                    t_ptr -> gradient[i] += (1.0f - tanhval * tanhval) * result_ptr -> gradient[i];
                }
            }, { t_ptr });
        }

        return result;
    }

    #pragma endregion

    #pragma region Loss Functions 

    std::shared_ptr<Tensor> softmax(const std::shared_ptr<Tensor>& t)
    {
        auto result = std::make_shared<Tensor>(t -> getShape(), 0.0f, t -> requiresGradient);

        const auto& x = t -> getData();
        auto& s = result -> getData();

        float maxV = *std::max_element(x.begin(), x.end());
        float sumExp = 0.0f;

        for (size_t i = 0; i < x.size(); ++i)
        {
            s[i] = std::exp(x[i] - maxV);
            sumExp += s[i];
        }

        for (size_t i = 0; i < s.size(); ++i)
            s[i] /= sumExp;

        if (t -> requiresGradient)
        {
            auto t_ptr = t -> shared_from_this();
            auto result_ptr = result;

            result->setGradientFunction([t_ptr, result_ptr]()
            {
                const auto& s = result_ptr -> data;
                const auto& gradOut = result_ptr -> gradient;
                auto& gradIn = t_ptr -> gradient;

                float dot = 0.0f;

                for (size_t j = 0; j < s.size(); ++j)
                    dot += gradOut[j] * s[j];

                for (size_t i = 0; i < s.size(); ++i)
                    gradIn[i] += s[i] * (gradOut[i] - dot);

            }, { t_ptr });
        }

        return result;
    }
        
    int argmax(const std::shared_ptr<Tensor>& t)
    {
        const auto& d = t -> getData();

        int best = 0;
        float maxV = d[0];

        for (int i = 1; i < (int)d.size(); ++i)
        {
            if (d[i] > maxV)
            {
                maxV = d[i];
                best = i;
            }
        }

        return best;
    }

    std::shared_ptr<Tensor> crossEntropyLoss(const std::shared_ptr<Tensor>& logits, const std::shared_ptr<Tensor>& targets)
    {
        assert(logits -> getShape() == targets -> getShape());

        const auto& x = logits -> getData();
        const auto& y = targets -> getData();

        size_t N = x.size();

        std::vector<float> s(N);

        float maxV = *std::max_element(x.begin(), x.end());
        float sumExp = 0.0f;

        for (size_t i = 0; i < N; ++i)
        {
            s[i] = std::exp(x[i] - maxV);
            sumExp += s[i];
        }

        for (size_t i = 0; i < N; ++i)
            s[i] /= sumExp;

        float loss = 0.0f;

        for (size_t i = 0; i < N; ++i)
            loss -= y[i] * std::log(s[i] + 1e-9f);

        loss /= static_cast<float>(N);

        auto result = std::make_shared<Tensor>(std::vector<int>{1}, loss, logits -> requiresGradient || targets -> requiresGradient);

        if (result -> requiresGradient)
        {
            auto log_ptr = logits -> shared_from_this();
            auto tgt_ptr = targets -> shared_from_this();
            auto res_ptr = result;

            result -> setGradientFunction([log_ptr, tgt_ptr, res_ptr, s, N]() mutable
            {
                float gradOut = res_ptr -> gradient[0] / static_cast<float>(N);
                auto& gradX = log_ptr -> gradient;
                const auto& yv = tgt_ptr -> getData();

                for (size_t i = 0; i < N; ++i)
                    gradX[i] += gradOut * (s[i] - yv[i]);

            }, { log_ptr, tgt_ptr });
        }

        return result;
    }

    #pragma endregion
}
