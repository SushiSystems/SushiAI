/**************************************************************************/
/* one_position_step.hpp                                                 */
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
 * @file one_position_step.hpp
 * @brief Declares OnePositionStep, the seam between CachedGenerator and the step it drives.
 * @author Mustafa Garip
 */

#pragma once

#include <string_view>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/language/generation/fed_leaves.hpp>
#include <SushiAI/language/generation/sampler.hpp>
#include <SushiAI/train/evaluation_step.hpp>

namespace SushiAI::Language::Generation
{
    /** @brief Runs one position of a model that keeps the keys and values of the ones before. */
    class OnePositionStep : public FedLeaves
    {
        public:
            /** @brief Reports whether the step keeps state between runs. */
            [[nodiscard]] virtual bool carries_state() const noexcept = 0;

            /**
             * @brief Returns the shape and the type of the output @p name.
             * @return The facts, or InvalidDescription when the step publishes no such output.
             */
            [[nodiscard]] virtual Result<LeafFacts> output(std::string_view name)
                const noexcept = 0;

            /**
             * @brief Returns the V scores of the output @p name from the last run.
             * @pre output() accepted @p name as float32 logits of [1, 1, V] and a run succeeded.
             */
            [[nodiscard]] virtual LogitSpan logits(std::string_view name) const noexcept = 0;

            /** @brief Clears the state the step keeps, so the next run is the first position. */
            [[nodiscard]] virtual Status reset() = 0;

            /** @brief Runs the step once; each feed is copied into the leaf it names. */
            [[nodiscard]] virtual Status run(SushiRuntime::span<const Train::NamedFeed> feeds) = 0;
    };

} // namespace SushiAI::Language::Generation
