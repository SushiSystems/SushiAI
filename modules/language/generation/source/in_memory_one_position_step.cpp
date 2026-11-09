/**************************************************************************/
/* in_memory_one_position_step.cpp                                       */
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
 * @file in_memory_one_position_step.cpp
 * @brief Implements InMemoryOnePositionStep.
 * @author Mustafa Garip
 */

#include <SushiAI/language/generation/in_memory_one_position_step.hpp>

#include <algorithm>
#include <string>
#include <utility>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/train/steps.hpp>

namespace SushiAI::Language::Generation
{
    InMemoryOnePositionStep::InMemoryOnePositionStep(std::vector<NamedLeafFacts> leaves,
                                                     std::string output,
                                                     std::int64_t vocabulary,
                                                     bool carries_state)
        : leaves_(std::move(leaves)),
          output_(std::move(output)),
          vocabulary_(vocabulary),
          carries_state_(carries_state),
          scores_(static_cast<std::size_t>(vocabulary), 0.0f)
    {

    }

    SushiRuntime::span<const std::string> InMemoryOnePositionStep::names() const noexcept
    {
        return leaves_.names();
    }

    Result<LeafFacts> InMemoryOnePositionStep::leaf(std::string_view name) const noexcept
    {
        return leaves_.leaf(name);
    }

    bool InMemoryOnePositionStep::carries_state() const noexcept
    {
        return carries_state_;
    }

    Result<LeafFacts> InMemoryOnePositionStep::output(std::string_view name) const noexcept
    {
        if (name != output_)
        {
            return Status::from_message(StatusCode::InvalidDescription, [&]()
            {
                return std::string("the step publishes no output '") + std::string(name) + "'";
            });
        }

        return LeafFacts{Shape{1, 1, vocabulary_}, DType::Float32};
    }

    LogitSpan InMemoryOnePositionStep::logits(std::string_view) const noexcept
    {
        return LogitSpan(scores_.data(), scores_.size());
    }

    Status InMemoryOnePositionStep::reset()
    {
        ++resets_;
        return Status();
    }

    Status InMemoryOnePositionStep::run(SushiRuntime::span<const Train::NamedFeed> feeds)
    {
        const Train::NamedFeed* const model_input =
            std::find_if(feeds.begin(), feeds.end(), [](const Train::NamedFeed& feed)
            {
                return feed.name == Train::K_MODEL_INPUT_NAME;
            });
        SA_FAIL_IF(model_input == feeds.end(), StatusCode::InvalidDescription,
                   "the run is fed no model input '{}'", Train::K_MODEL_INPUT_NAME);

        const std::int32_t id = model_input->tensor->data_as<std::int32_t>()[0];
        fed_ids_.push_back(id);
        std::fill(scores_.begin(), scores_.end(), 0.0f);
        scores_[static_cast<std::size_t>((id + 1) % vocabulary_)] = 1.0f;

        return Status();
    }

    const std::vector<std::int32_t>& InMemoryOnePositionStep::fed_ids() const noexcept
    {
        return fed_ids_;
    }

    std::size_t InMemoryOnePositionStep::resets() const noexcept
    {
        return resets_;
    }

} // namespace SushiAI::Language::Generation
