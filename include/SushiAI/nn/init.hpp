/**************************************************************************/
/* init.hpp                                                               */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                https://github.com/SushiSystems/SushiAI                 */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#pragma once

/**
 * @file init.hpp
 * @brief Where a model's weights start, and why.
 *
 * @section gains The gain table
 *
 * Both families below scale a zero-mean distribution so the variance of a
 * layer's output matches the variance of its input. The scale depends on the
 * nonlinearity that follows, because an activation that discards half its
 * input halves the variance:
 *
 * | nonlinearity | gain                |
 * |--------------|---------------------|
 * | linear       | 1                   |
 * | sigmoid      | 1                   |
 * | tanh         | 5/3                 |
 * | relu         | sqrt(2)             |
 * | leaky relu   | sqrt(2 / (1 + a^2)) |
 *
 * Kaiming scales by fan-in alone (right for a rectifier, where only the
 * forward variance is preserved anyway); Xavier scales by both fans (right for
 * a symmetric activation, where the backward variance matters equally).
 *
 * @section pytorch A deliberate divergence from PyTorch's default
 *
 * @c torch.nn.Linear does not use any of the above. Its default is
 * @c kaiming_uniform_(a=sqrt(5)), which yields gain = sqrt(2/(1+5)) = 1/sqrt(3)
 * and therefore a variance 6x smaller than He initialisation for the ReLU
 * networks it is usually paired with. It is a historical accident of the Torch7
 * port, documented as such in PyTorch's own issue tracker, and it makes deep
 * ReLU stacks start with activations that shrink layer over layer. SushiAI
 * uses the gain the nonlinearity actually calls for. A model ported from
 * PyTorch will therefore not match its reference initialisation bit for bit -
 * that is intended, and it is stated here so nobody spends an afternoon
 * hunting for the difference.
 *
 * @section final Zero-initialised classifier
 *
 * The last weight matrix (and its bias) start at exactly zero. Every logit is
 * then zero, every class is equally likely, and the initial cross-entropy is
 * exactly ln(C) - a number that can be asserted rather than eyeballed. It also
 * removes the initial random logit spread, which is pure noise in the first
 * few steps. The layer leaves zero immediately: its gradient is
 * (softmax - target) / N, which is non-zero from the first step.
 */

#include <cstdint>
#include <string>
#include <SushiBLAS/engine.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/graph/lowering.hpp>
#include <SushiAI/nn/module.hpp>

namespace SushiAI
{
    namespace NN
    {
        /**
         * @enum Nonlinearity
         * @brief The activation a layer feeds, which sets its initialisation gain.
         */
        enum class Nonlinearity : std::uint8_t
        {
            LINEAR,     /**< No activation, or an identity one. */
            TANH,       /**< Hyperbolic tangent. */
            RELU,       /**< Rectified linear unit; also GELU and SiLU in practice. */
            LEAKY_RELU, /**< Leaky rectified linear unit. */
            SIGMOID,    /**< Logistic sigmoid. */
        };

        /**
         * @brief The variance-preserving gain for @p nonlinearity.
         * @param nonlinearity Which activation follows the layer.
         * @param negative_slope The leaky-ReLU slope; ignored otherwise.
         * @return The multiplicative gain; always positive.
         */
        double calculate_gain(Nonlinearity nonlinearity, double negative_slope = 0.01) noexcept;

        /**
         * @brief Fills @p tensor from N(0, (gain / sqrt(fan_in))^2).
         * @param engine The engine whose Philox stream supplies the numbers.
         * @param tensor The tensor to fill.
         * @param fan_in Inputs feeding one output unit.
         * @param gain The nonlinearity's gain.
         */
        void kaiming_normal(SushiBLAS::Engine& engine, SushiBLAS::Tensor& tensor,
                            int64_t fan_in, double gain);

        /**
         * @brief Fills @p tensor uniformly on [-b, b] with b = gain * sqrt(3 / fan_in).
         * @param engine The engine whose Philox stream supplies the numbers.
         * @param tensor The tensor to fill.
         * @param fan_in Inputs feeding one output unit.
         * @param gain The nonlinearity's gain.
         */
        void kaiming_uniform(SushiBLAS::Engine& engine, SushiBLAS::Tensor& tensor,
                             int64_t fan_in, double gain);

        /**
         * @brief Fills @p tensor from N(0, (gain * sqrt(2 / (fan_in + fan_out)))^2).
         * @param engine The engine whose Philox stream supplies the numbers.
         * @param tensor The tensor to fill.
         * @param fan_in Inputs feeding one output unit.
         * @param fan_out Outputs one input unit feeds.
         * @param gain The nonlinearity's gain.
         */
        void xavier_normal(SushiBLAS::Engine& engine, SushiBLAS::Tensor& tensor,
                           int64_t fan_in, int64_t fan_out, double gain);

        /**
         * @brief Fills @p tensor uniformly on [-a, a] with a = gain * sqrt(6 / (fan_in + fan_out)).
         * @param engine The engine whose Philox stream supplies the numbers.
         * @param tensor The tensor to fill.
         * @param fan_in Inputs feeding one output unit.
         * @param fan_out Outputs one input unit feeds.
         * @param gain The nonlinearity's gain.
         */
        void xavier_uniform(SushiBLAS::Engine& engine, SushiBLAS::Tensor& tensor,
                            int64_t fan_in, int64_t fan_out, double gain);

        /**
         * @brief Fills @p tensor with zeros.
         * @param engine The engine to record the fill on.
         * @param tensor The tensor to zero.
         */
        void zeros(SushiBLAS::Engine& engine, SushiBLAS::Tensor& tensor);

        /**
         * @enum InitFamily
         * @brief Which variance rule to apply to weights.
         */
        enum class InitFamily : std::uint8_t
        {
            KAIMING, /**< Scale by fan-in; the right default for a rectifier stack. */
            XAVIER,  /**< Scale by both fans; the right default for tanh or sigmoid. */
        };

        /**
         * @struct InitPolicy
         * @brief How a whole model is initialised.
         */
        struct InitPolicy
        {
            /** @brief Which variance rule the weights follow. */
            InitFamily family = InitFamily::KAIMING;

            /** @brief The activation the hidden layers feed. */
            Nonlinearity nonlinearity = Nonlinearity::RELU;

            /** @brief Draw uniformly rather than normally. */
            bool uniform = false;

            /** @brief The leaky-ReLU slope, when that is the nonlinearity. */
            double negative_slope = 0.01;

            /** @brief Zero the last weight and bias, so the initial loss is exactly ln(C). */
            bool zero_final_layer = true;

            /** @brief The Philox seed; the same seed gives the same weights. */
            std::uint64_t seed = 0x51757368'69414921ull;
        };

        namespace Detail
        {
            /**
             * @brief Applies @p policy to one parameter's tensor.
             *
             * Split out of the template below so the arithmetic lives in one
             * translation unit rather than in every consumer's.
             *
             * @param engine The engine to record fills on.
             * @param tensor The parameter's tensor.
             * @param parameter The parameter's description.
             * @param policy The initialisation policy.
             * @param is_final_layer Whether this belongs to the zeroed classifier.
             */
            void apply_initializer(SushiBLAS::Engine& engine,
                                   SushiBLAS::Tensor& tensor,
                                   const Parameter& parameter,
                                   const InitPolicy& policy,
                                   bool is_final_layer);
        }

        /**
         * @brief Initialises every parameter of @p module and waits for the device.
         *
         * Seeds the engine's Philox stream from @p policy so a run is
         * reproducible, walks the parameters through the one generic walk (so
         * a layer cannot be skipped - see module.hpp), and flushes.
         *
         * The "final layer" is the last @c WEIGHT the walk yields, together
         * with everything after it. That is a heuristic, and a deliberate one:
         * it is exactly right for a feed-forward classifier, which is what
         * @c zero_final_layer exists for, and it is stated here rather than
         * inferred from a layer type the walk cannot see.
         *
         * @param engine The engine that owns the parameter tensors' memory.
         * @param module The model to initialise; must already have been traced.
         * @param pool Where the parameters' tensors live.
         * @param policy The initialisation policy.
         * @throws Error If a parameter has no tensor bound in @p pool.
         */
        template<typename M>
        void initialize(SushiBLAS::Engine& engine, M& module, Graph::TensorPool& pool,
                        const InitPolicy& policy = {})
        {
            engine.set_seed(policy.seed);

            std::vector<Parameter*> parameters;
            for_each_parameter(module, [&parameters](const std::string&, Parameter& parameter)
            {
                parameters.push_back(&parameter);
            });

            std::size_t final_weight = parameters.size();
            for (std::size_t i = 0; i < parameters.size(); ++i)
                if (parameters[i]->role() == ParameterRole::WEIGHT) final_weight = i;

            for (std::size_t i = 0; i < parameters.size(); ++i)
            {
                const bool is_final = policy.zero_final_layer && i >= final_weight;
                Detail::apply_initializer(engine, pool.tensor(parameters[i]->id()),
                                          *parameters[i], policy, is_final);
            }

            engine.execute();
            engine.get_context().wait_all();
        }

    } // namespace NN
} // namespace SushiAI
