/**************************************************************************/
/*  initializer.h                                                         */
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

#pragma once
#include <cmath>
#include <random>
#include <algorithm>

#include "tensor.h"

namespace SushiAI 
{
    #pragma region Initializer Class

    /// Abstract base class for all weight initializers.
    class Initializer 
    {
        public:
            virtual ~Initializer() = default;

            virtual void initialize(const std::shared_ptr<Tensor>& t) const = 0;
    };

    #pragma endregion

    #pragma region Initializers

    /// Initializes tensor values with a uniform distribution in range [a, b].
    class UniformInitializer : public Initializer
    {
        float lower, upper;

        public:
            UniformInitializer(float a, float b) : lower(a), upper(b) {}

            void initialize(const std::shared_ptr<Tensor>& t) const override
            {
                std::mt19937 gen(std::random_device{}());
                std::uniform_real_distribution<float> dist(lower, upper);

                auto& ptr = t -> getData();
                size_t N = t -> getTotalSize();

                for (size_t i = 0; i < N; ++i)
                    ptr[i] = dist(gen);
            }
        };

    /// Initializes tensor values with a normal (Gaussian) distribution.
    class NormalInitializer : public Initializer
    {
        float mean, stddev;

        public:
            NormalInitializer(float m, float s) : mean(m), stddev(s) {}

            void initialize(const std::shared_ptr<Tensor>& t) const override
            {
                std::mt19937 gen(std::random_device{}());
                std::normal_distribution<float> dist(mean, stddev);

                auto& ptr = t -> getData();
                size_t N = t -> getTotalSize();

                for (size_t i = 0; i < N; ++i)
                    ptr[i] = dist(gen);
            }
    };

    /// Computes fan-in and fan-out for a tensor based on its shape.
    inline void computeFans(const std::shared_ptr<Tensor>& t, int& fan_in, int& fan_out)
    {
        auto shape = t -> getShape();

        if (shape.size() == 2)
        {
            fan_in = shape[0];
            fan_out = shape[1];
        }
        else if (shape.size() > 2)
        {
            // Conv: [C_out, C_in, kH, kW]
            fan_in = shape[1] * shape[2] * shape[3];
            fan_out = shape[0] * shape[2] * shape[3];
        }
        else
            fan_in = fan_out = 1;
    }

    /// Xavier (Glorot) uniform initialization. Recommended for Tanh/Sigmoid activations.
    class XavierUniform : public Initializer
    {
        public:
            void initialize(const std::shared_ptr<Tensor>& t) const override
            {
                int fan_in, fan_out;
                computeFans(t, fan_in, fan_out);

                float bound = std::sqrt(6.0f / (fan_in + fan_out));
                UniformInitializer u(-bound, bound);
                u.initialize(t);
            }
    };

    /// Xavier (Glorot) normal initialization. Also suitable for Tanh/Sigmoid activations.
    class XavierNormal : public Initializer
    {
        public:
            void initialize(const std::shared_ptr<Tensor>& t) const override
            {
                int fan_in, fan_out;
                computeFans(t, fan_in, fan_out);

                float stddev = std::sqrt(2.0f / (fan_in + fan_out));
                NormalInitializer n(0.0f, stddev);
                n.initialize(t);
            }
    };

    /// He (Kaiming) uniform initialization. Recommended for ReLU or variants.
    class HeUniform : public Initializer
    {
        public:
            void initialize(const std::shared_ptr<Tensor>& t) const override
            {
                int fan_in, fan_out;
                computeFans(t, fan_in, fan_out);

                float bound = std::sqrt(6.0f / fan_in);
                UniformInitializer u(-bound, bound);
                u.initialize(t);
            }
    };

    /// He (Kaiming) normal initialization. Also suitable for ReLU or variants.
    class HeNormal : public Initializer
    {
        public:
            void initialize(const std::shared_ptr<Tensor>& t) const override
            {
                int fan_in, fan_out;
                computeFans(t, fan_in, fan_out);

                float stddev = std::sqrt(2.0f / fan_in);
                NormalInitializer n(0.0f, stddev);
                n.initialize(t);
            }
    };

    /// LeCun uniform initialization. Recommended for self-normalizing networks (e.g., with SELU).
    class LeCunUniform : public Initializer
    {
        public:
            void initialize(const std::shared_ptr<Tensor>& t) const override
            {
                int fan_in, fan_out;
                computeFans(t, fan_in, fan_out);

                float bound = std::sqrt(3.0f / fan_in);
                UniformInitializer u(-bound, bound);
                u.initialize(t);
            }
    };

    #pragma endregion
}
