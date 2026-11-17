/**************************************************************************/
/* objective_output.cpp                                                   */
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
 * @file objective_output.cpp
 * @brief Implements the selector of the output an objective reads and the reader of its keys.
 * @author Mustafa Garip
 */

#include <SushiAI/train/objective_output.hpp>

#include <cstddef>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/seam/seam.hpp>

namespace SushiAI
{
    namespace Train
    {
        Result<ObjectiveOutput> ObjectiveOutput::named(std::string_view name) noexcept
        {
            SA_FAIL_IF(name.empty(), StatusCode::InvalidDescription,
                       "An objective names its output with a string of at least 1 character");

            return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                ObjectiveOutput made;
                made.name_ = std::string(name);
                return made;
            });
        }

        Result<ObjectiveOutput> ObjectiveOutput::read(const Config::Value& attributes,
                                                      std::string_view context) noexcept
        {
            if (!attributes.has(K_OUTPUT_KEY))
                return ObjectiveOutput();

            SA_TRY_ASSIGN(const Config::Value& stated, attributes.at(K_OUTPUT_KEY, context));
            SA_FAIL_IF(stated.kind() != Config::Kind::String, StatusCode::InvalidDescription,
                       "{}: '{}' is the name of a model output, but the file holds a {}",
                       context, K_OUTPUT_KEY, Config::kind_name(stated.kind()));
            SA_TRY_ASSIGN(const std::string& name, stated.as_string(context));

            return named(name);
        }

        Result<ObjectiveTarget> ObjectiveTarget::named(std::string_view role) noexcept
        {
            SA_FAIL_IF(role.empty(), StatusCode::InvalidDescription,
                       "An objective names its target role with a string of at least 1 "
                       "character");

            return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                ObjectiveTarget made;
                made.role_ = std::string(role);
                return made;
            });
        }

        Result<ObjectiveTarget> ObjectiveTarget::read(const Config::Value& attributes,
                                                      std::string_view context) noexcept
        {
            if (!attributes.has(K_TARGET_KEY))
                return ObjectiveTarget();

            SA_TRY_ASSIGN(const Config::Value& stated, attributes.at(K_TARGET_KEY, context));
            SA_FAIL_IF(stated.kind() != Config::Kind::String, StatusCode::InvalidDescription,
                       "{}: '{}' is the name of a batch role, but the file holds a {}",
                       context, K_TARGET_KEY, Config::kind_name(stated.kind()));
            SA_TRY_ASSIGN(const std::string& role, stated.as_string(context));

            return named(role);
        }

        Result<Graph::ValueID> ObjectiveOutput::resolve(
            const NN::ModelOutputs& outputs) const noexcept
        {
            if (!is_named())
                return outputs.only();

            return outputs.at(name_);
        }

        Result<const SushiBLAS::Tensor*> ObjectiveOutput::resolve(
            const StepView& step) const noexcept
        {
            if (!is_named())
                return only_output(step);

            const StepOutput* found = nullptr;
            for (const StepOutput& output : step.outputs)
            {
                if (output.name == name_)
                    found = &output;
            }

            SA_FAIL_IF(found == nullptr, StatusCode::InvalidDescription,
                       "The step published {} outputs and none is named '{}'",
                       step.outputs.size(), name_);
            SA_FAIL_IF(found->tensor == nullptr, StatusCode::InvalidDescription,
                       "The output '{}' of the step has no tensor", name_);

            return found->tensor;
        }

        Result<const OutputShape*> ObjectiveOutput::resolve(
            const ObjectiveContext& given, std::string_view context) const noexcept
        {
            if (!is_named())
            {
                SA_FAIL_IF(given.outputs.size() != 1, StatusCode::InvalidDescription,
                           "{} reads the model's only output, but the model publishes {}; "
                           "name one with '{}'",
                           context, given.outputs.size(), K_OUTPUT_KEY);

                return &given.outputs.front();
            }

            const OutputShape* const found = find_output_shape(given.outputs, name_);
            SA_FAIL_IF(found == nullptr, StatusCode::InvalidDescription,
                       "{} reads the output '{}', but the model publishes {} outputs and "
                       "none has that name",
                       context, name_, given.outputs.size());

            return found;
        }

        bool is_selection_key(std::string_view key) noexcept
        {
            return key == K_OUTPUT_KEY || key == K_TYPE_MASK_KEY || key == K_TARGET_KEY;
        }

        Result<ObjectiveSelection> read_selection(const Config::Value& attributes,
                                                  std::string_view context,
                                                  std::string_view own_key) noexcept
        {
            for (const std::string& key : attributes.keys())
            {
                if (is_selection_key(key) || (!own_key.empty() && key == own_key))
                    continue;

                SA_FAIL_IF(own_key.empty(), StatusCode::InvalidDescription,
                           "{} states the key '{}', but the only keys it reads are '{}', '{}' "
                           "and '{}'",
                           context, key, K_OUTPUT_KEY, K_TYPE_MASK_KEY, K_TARGET_KEY);
                SA_FAIL_IF(!own_key.empty(), StatusCode::InvalidDescription,
                           "{} states the key '{}', but the only keys it reads are '{}', '{}', "
                           "'{}' and '{}'",
                           context, key, own_key, K_OUTPUT_KEY, K_TYPE_MASK_KEY, K_TARGET_KEY);
            }

            ObjectiveSelection selection;
            SA_TRY_ASSIGN(selection.output, ObjectiveOutput::read(attributes, context));
            SA_TRY_ASSIGN(selection.mask, RowMask::read(attributes, context));
            SA_TRY_ASSIGN(selection.target, ObjectiveTarget::read(attributes, context));

            return selection;
        }

    } // namespace Train
} // namespace SushiAI
