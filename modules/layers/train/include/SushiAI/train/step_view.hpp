/**************************************************************************/
/* step_view.hpp                                                          */
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
 * @file step_view.hpp
 * @brief Declares what an objective reads of a step that has run.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <string_view>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/data/dataset.hpp>

namespace SushiAI
{
    namespace Train
    {
        /**
         * @brief Pairs the name of a published value with the tensor a step wrote: a model
         *        output, or a value an objective declared with Objective::declare_metric_values().
         */
        struct StepOutput
        {
            std::string name;                          /**< Name it is published under. */
            const SushiBLAS::Tensor* tensor = nullptr; /**< Its tensor; the pool owns it. */
        };

        /** @brief Holds what an objective reads of a step that has run. */
        struct StepView
        {
            const Data::Batch& batch;                     /**< The batch the step trained on. */
            double loss = 0.0;                            /**< The loss the step computed. */
            SushiRuntime::span<const StepOutput> outputs; /**< Outputs, in published order. */
            SushiRuntime::span<const StepOutput> values{}; /**< Objective values, declared order. */
        };

        /**
         * @brief Returns the tensor of a step that published exactly one output.
         * @pre No recorded task writes the tensor.
         * @return The tensor, never null, or InvalidDescription stating how many outputs
         *         the step published.
         */
        [[nodiscard]] Result<const SushiBLAS::Tensor*> only_output(const StepView& step) noexcept;

        /**
         * @brief Returns the tensor of the value an objective declared under @p name.
         * @pre No recorded task writes the tensor.
         * @return The tensor, never null, or InvalidDescription naming @p name and the values
         *         the step holds.
         */
        [[nodiscard]] Result<const SushiBLAS::Tensor*> named_value(const StepView& step,
                                                                   std::string_view name) noexcept;

    } // namespace Train
} // namespace SushiAI
