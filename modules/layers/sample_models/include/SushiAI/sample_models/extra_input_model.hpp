/**************************************************************************/
/* extra_input_model.hpp                                                  */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file extra_input_model.hpp
 * @brief Declares the model of one extra input the tests of a trainer and an evaluator build.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/nn/add.hpp>
#include <SushiAI/nn/flatten.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/shape/shape.hpp>

namespace SushiAI
{
    namespace Test
    {
        /** @brief Names the extra input of build_extra_input_model(). */
        inline constexpr std::string_view K_EXTRA_INPUT_OFFSET = "offset";

        /** @brief Counts the elements of one sample of the input and of the extra input. */
        inline constexpr std::int64_t K_EXTRA_MODEL_ELEMENTS = 12;

        /** @brief Counts the classes the model ends in. */
        inline constexpr std::int64_t K_EXTRA_MODEL_CLASSES = 3;

        /** @brief Returns the per-sample shape of the input and of the extra input: [2, 2, 3]. */
        [[nodiscard]] inline Shape extra_model_sample() noexcept
        {
            return Shape{2, 2, 3};
        }

        /**
         * @brief Builds the sum of the [2, 2, 3] input and the [2, 2, 3] extra input "offset",
         *        then Flatten and Linear{12, 3}; publishes "logits".
         */
        [[nodiscard]] inline Result<NN::ModelGraph> build_extra_input_model() noexcept
        {
            NN::ModelGraph model(extra_model_sample());

            SA_TRY_ASSIGN(const NN::NodeRef offset,
                          model.declare_input(std::string(K_EXTRA_INPUT_OFFSET),
                                              extra_model_sample(), DType::Float32));
            const NN::NodeRef both[] = {model.input(), offset};
            const Shape shapes[] = {model.shape_of(model.input()), model.shape_of(offset)};
            SA_TRY_ASSIGN(std::unique_ptr<NN::Add> add, NN::Add::create(NN::ShapeSpan(shapes, 2)));
            SA_TRY_ASSIGN(const NN::NodeRef sum, model.add("sum", std::move(add), both));
            SA_TRY_ASSIGN(const NN::NodeRef flat, model.add("flat", NN::Flatten{}, sum));
            SA_TRY_ASSIGN(const NN::NodeRef logits,
                          model.add("head", NN::Linear{K_EXTRA_MODEL_ELEMENTS,
                                                       K_EXTRA_MODEL_CLASSES}, flat));
            SA_TRY(model.output("logits", logits));

            return model;
        }

    } // namespace Test
} // namespace SushiAI
