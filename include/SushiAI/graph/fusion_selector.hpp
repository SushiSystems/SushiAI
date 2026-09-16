/**************************************************************************/
/* fusion_selector.hpp                                                    */
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

/**
 * @file fusion_selector.hpp
 * @brief Declares the dynamic fusion selector interface and device policy.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <string_view>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/engine/device_policy.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/graph/fusion.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /**
         * @struct FusionQuery
         * @brief One fused node, described in the terms a selector reasons in.
         */
        struct FusionQuery
        {
            /** @brief The form the static pass matched; never @c FusedForm::NONE. */
            FusedForm matched = FusedForm::NONE;

            /** @brief The element type every operand shares. */
            DType dtype = DType::FLOAT32;

            /** @brief Rows of the node's result. */
            std::int64_t rows = 0;

            /** @brief Columns of the node's result. */
            std::int64_t columns = 0;

            /**
             * @brief Whether the epilogue's broadcast operand has unit stride.
             *
             * A bias epilogue indexes @c bias[col] directly, so a strided bias
             * would be read at the wrong elements. Meaningless for a form with
             * no epilogue operand, and true for those.
             */
            bool operand_is_contiguous = true;
        };

        /**
         * @struct FusionChoice
         * @brief What the lowering should spell, and why.
         */
        struct FusionChoice
        {
            /**
             * @brief The form to instantiate.
             *
             * Equal to @c FusionQuery::matched to take the whole match, a
             * shorter entry to take part of it, or @c FusedForm::NONE to spell
             * the node out of unfused operations.
             */
            FusedForm form = FusedForm::NONE;

            /** @brief A sentence for the report; a static string, never owned. */
            std::string_view reason{};
        };

        /**
         * @class IFusionSelector
         * @brief Defines the interface for bind-time selection of fused forms.
         */
        class IFusionSelector
        {
            public:
                virtual ~IFusionSelector() = default;

                /**
                 * @brief Chooses the form to spell for @p query.
                 * @param query The matched node.
                 * @return The chosen form and the reason for it.
                 */
                virtual FusionChoice choose(const FusionQuery& query) const = 0;
        };

        /**
         * @class DeviceFusionSelector
         * @brief Selects fused forms based on device profile and execution mode.
         */
        class DeviceFusionSelector final : public IFusionSelector
        {
            public:
                /**
                 * @brief Constructs a selector with explicit profile and execution mode.
                 * @param profile The device profile.
                 * @param mode The engine execution mode.
                 */
                DeviceFusionSelector(const SushiBLAS::Device::Profile& profile,
                                     SushiBLAS::ExecutionMode mode) noexcept
                    : profile_(profile), mode_(mode) {}

                /**
                 * @brief Builds a selector describing @p engine's real device.
                 * @param engine The engine the step will be recorded on.
                 * @return A selector over that device's cached profile.
                 */
                static DeviceFusionSelector for_engine(SushiBLAS::Engine& engine);

                /** @brief The device this selector is reasoning about. */
                const SushiBLAS::Device::Profile& profile() const noexcept { return profile_; }

                /** @brief The execution mode this selector is reasoning about. */
                SushiBLAS::ExecutionMode mode() const noexcept { return mode_; }

                FusionChoice choose(const FusionQuery& query) const override;

            private:
                SushiBLAS::Device::Profile profile_;
                SushiBLAS::ExecutionMode mode_;
        };

        /**
         * @struct FusionSelection
         * @brief One selector decision, kept so the lowering can be interrogated.
         */
        struct FusionSelection
        {
            /** @brief What the static pass matched. */
            FusedForm matched = FusedForm::NONE;

            /** @brief What the selector chose. */
            FusedForm chosen = FusedForm::NONE;

            /** @brief Why. */
            std::string_view reason{};

            /** @brief How many device tasks the chosen spelling recorded. */
            std::size_t tasks = 0;
        };

    } // namespace Graph
} // namespace SushiAI
