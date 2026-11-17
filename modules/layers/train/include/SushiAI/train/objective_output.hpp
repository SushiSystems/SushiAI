/**************************************************************************/
/* objective_output.hpp                                                   */
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
 * @file objective_output.hpp
 * @brief Declares what an objective section selects: one model output and a row mask.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <string_view>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/train/row_mask.hpp>
#include <SushiAI/train/step_view.hpp>

namespace SushiAI
{
    namespace Train
    {
        /** @brief Names the key of an objective section that names the output it reads. */
        inline constexpr std::string_view K_OUTPUT_KEY = "output";

        /** @brief Names the key of an objective section that names the batch role it reads. */
        inline constexpr std::string_view K_TARGET_KEY = "target";

        /**
         * @brief Names the model output an objective reads, in the graph, in a step and in
         *        the context a factory sees.
         *
         * The default reads the model's only output. A named selector reads the port of that
         * name and leaves every other port of the model alone.
         */
        class ObjectiveOutput final
        {
            public:
                /** @brief Builds the selector of the model's only output. */
                ObjectiveOutput() noexcept = default;

                /**
                 * @brief Builds the selector of the output published as @p name.
                 * @return The selector, or InvalidDescription for an empty name.
                 */
                [[nodiscard]] static Result<ObjectiveOutput> named(std::string_view name) noexcept;

                /**
                 * @brief Reads the optional K_OUTPUT_KEY of an objective section.
                 * @param attributes The keys of the objective.
                 * @param context What to call the objective in a diagnostic.
                 * @return The selector; the default one when the key is absent; or
                 *         InvalidDescription for a value that is not a non-empty string.
                 */
                [[nodiscard]] static Result<ObjectiveOutput> read(const Config::Value& attributes,
                                                                  std::string_view context) noexcept;

                /** @brief Returns true when an output is named. */
                [[nodiscard]] bool is_named() const noexcept { return !name_.empty(); }

                /** @brief Returns the name of the output; empty for the default selector. */
                [[nodiscard]] const std::string& name() const noexcept { return name_; }

                /**
                 * @brief Returns the value of the selected output.
                 * @return The value, or InvalidDescription when the model publishes no such
                 *         output, or, for the default selector, not exactly one.
                 */
                [[nodiscard]] Result<Graph::ValueID> resolve(
                    const NN::ModelOutputs& outputs) const noexcept;

                /**
                 * @brief Returns the tensor a step wrote for the selected output.
                 * @pre No recorded task writes the tensor.
                 * @return The tensor, never null, or InvalidDescription when the step
                 *         published no such output, or, for the default selector, not
                 *         exactly one.
                 */
                [[nodiscard]] Result<const SushiBLAS::Tensor*> resolve(
                    const StepView& step) const noexcept;

                /**
                 * @brief Returns the entry of the selected output in @p given.
                 * @param context What to call the objective in a diagnostic.
                 * @return A pointer into @p given, never null, or InvalidDescription when the
                 *         model publishes no such output, or, for the default selector, not
                 *         exactly one.
                 */
                [[nodiscard]] Result<const OutputShape*> resolve(
                    const ObjectiveContext& given, std::string_view context) const noexcept;

            private:
                /// @brief The name of the output; empty reads the only one.
                std::string name_;
        };

        /**
         * @brief Names the batch role an objective reads its targets from.
         *
         * The default reads the role the objective is written for, such as "labels" or "values".
         * A named selector reads the role of that name instead, so one batch can carry the
         * targets of several objectives under several roles.
         */
        class ObjectiveTarget final
        {
            public:
                /** @brief Builds the selector of the role the objective is written for. */
                ObjectiveTarget() noexcept = default;

                /**
                 * @brief Builds the selector of the role @p role.
                 * @return The selector, or InvalidDescription for an empty name.
                 */
                [[nodiscard]] static Result<ObjectiveTarget> named(std::string_view role) noexcept;

                /**
                 * @brief Reads the optional K_TARGET_KEY of an objective section.
                 * @param attributes The keys of the objective.
                 * @param context What to call the objective in a diagnostic.
                 * @return The selector; the default one when the key is absent; or
                 *         InvalidDescription for a value that is not a non-empty string.
                 */
                [[nodiscard]] static Result<ObjectiveTarget> read(const Config::Value& attributes,
                                                                  std::string_view context) noexcept;

                /** @brief Returns true when a role is named. */
                [[nodiscard]] bool is_named() const noexcept { return !role_.empty(); }

                /**
                 * @brief Returns the role the objective reads.
                 * @param fallback The role the objective is written for.
                 * @return The named role, or @p fallback for the default selector.
                 */
                [[nodiscard]] std::string_view role_or(std::string_view fallback) const noexcept
                {
                    return is_named() ? std::string_view(role_) : fallback;
                }

            private:
                /// @brief The name of the role; empty reads the role the objective is written for.
                std::string role_;
        };

        /** @brief Holds what the optional keys of an objective section select. */
        struct ObjectiveSelection
        {
            ObjectiveOutput output; /**< The output the objective reads. */
            RowMask mask;           /**< The rows the loss keeps. */
            ObjectiveTarget target; /**< The batch role the objective reads its targets from. */
        };

        /**
         * @brief Returns true for K_OUTPUT_KEY, K_TYPE_MASK_KEY and K_TARGET_KEY, the keys
         *        read_selection() reads.
         */
        [[nodiscard]] bool is_selection_key(std::string_view key) noexcept;

        /**
         * @brief Reads the selection keys of an objective section and refuses any other key.
         * @param context What to call the objective in a diagnostic.
         * @param own_key The one key the objective reads itself, e.g. "threshold"; empty for none.
         * @return The selection, or InvalidDescription for a key that is neither a selection
         *         key nor @p own_key, or for what the readers of the three keys refuse.
         */
        [[nodiscard]] Result<ObjectiveSelection> read_selection(
            const Config::Value& attributes, std::string_view context,
            std::string_view own_key = {}) noexcept;

    } // namespace Train
} // namespace SushiAI
