/**************************************************************************/
/* role_match.cpp                                                         */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file role_match.cpp
 * @brief Implements the check of the roles a step reads against the roles a dataset publishes.
 * @author Mustafa Garip
 */

#include <SushiAI/train/role_match.hpp>

#include <string>
#include <vector>
#include <SushiLog/text/text_format.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/target.hpp>

namespace SushiAI
{
    namespace Train
    {
        namespace
        {
            /** @brief Renders @p target as "4 classes of int32 [] per sample". */
            std::string describe(const Data::TargetDescription& target)
            {
                const std::string values = SushiLog::format_text(
                    "{} {} per sample", dtype_name(target.dtype), target.sample_shape.to_string());

                return target.classes == 0
                     ? values
                     : SushiLog::format_text("{} classes of {}", target.classes, values);
            }

            /** @brief Lists the roles of @p targets in quotes, or "(none)". */
            std::string list_roles(const std::vector<Data::TargetDescription>& targets)
            {
                std::string listed;
                for (const Data::TargetDescription& target : targets)
                {
                    if (!listed.empty())
                        listed += ", ";
                    listed += "'" + target.role + "'";
                }

                return listed.empty() ? std::string("(none)") : listed;
            }

            /** @brief Returns whether @p published serves a step that reads @p required. */
            bool is_served_by(const Data::TargetDescription& required,
                              const Data::TargetDescription& published) noexcept
            {
                return required.sample_shape == published.sample_shape
                    && required.dtype == published.dtype
                    && (required.classes == 0 || required.classes == published.classes);
            }

            /** @brief Checks one role of a step against @p dataset. */
            [[nodiscard]] Status match_role(const Data::TargetDescription& required,
                                            const Data::Dataset& dataset)
            {
                const Data::TargetDescription* published =
                    Data::find_target(dataset.targets(), required.role);

                SA_FAIL_IF(published == nullptr, StatusCode::InvalidDescription,
                           "The dataset '{}' does not publish the role '{}', which the step "
                           "reads as {}; it publishes {}",
                           dataset.name(), required.role, describe(required),
                           list_roles(dataset.targets()));
                SA_FAIL_IF(!is_served_by(required, *published), StatusCode::Incompatible,
                           "The dataset '{}' publishes the role '{}' as {}, but the step was "
                           "compiled for {}",
                           dataset.name(), required.role, describe(*published),
                           describe(required));

                return Status();
            }
        } // namespace

        Status match_roles(const std::vector<Data::TargetDescription>& required,
                           const Data::Dataset& dataset) noexcept
        {
            return ::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Allocate,
                                            [&required, &dataset]() -> Status
            {
                for (const Data::TargetDescription& role : required)
                    SA_TRY(match_role(role, dataset));

                return Status();
            });
        }

    } // namespace Train
} // namespace SushiAI
