/**************************************************************************/
/* step_view.cpp                                                          */
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
 * @file step_view.cpp
 * @brief Implements the lookup of the one output of a step that has run.
 * @author Mustafa Garip
 */

#include <SushiAI/train/step_view.hpp>

#include <string>
#include <SushiLog/text/text_format.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Train
    {
        Result<const SushiBLAS::Tensor*> only_output(const StepView& step) noexcept
        {
            SA_FAIL_IF(step.outputs.size() != 1, StatusCode::InvalidDescription,
                       "The metric reads one model output and the step published {}",
                       step.outputs.size());
            SA_FAIL_IF(step.outputs[0].tensor == nullptr, StatusCode::InvalidDescription,
                       "The output '{}' of the step has no tensor", step.outputs[0].name);

            return step.outputs[0].tensor;
        }

        Result<const SushiBLAS::Tensor*> named_value(const StepView& step,
                                                     std::string_view name) noexcept
        {
            for (const StepOutput& value : step.values)
            {
                if (value.name != name)
                    continue;

                SA_FAIL_IF(value.tensor == nullptr, StatusCode::InvalidDescription,
                           "The value '{}' of the step has no tensor", name);

                return value.tensor;
            }

            return Status::from_message(StatusCode::InvalidDescription, [&]()
            {
                std::string held;
                for (const StepOutput& value : step.values)
                    held += (held.empty() ? "'" : ", '") + value.name + "'";

                return SushiLog::format_text(
                    "The step holds no objective value '{}'; it holds: {}", name,
                    held.empty() ? std::string("(none)") : held);
            });
        }

    } // namespace Train
} // namespace SushiAI
