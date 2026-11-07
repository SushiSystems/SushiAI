/**************************************************************************/
/* fusion_selector.hpp                                                    */
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
 * @file fusion_selector.hpp
 * @brief Declares the dynamic fusion selector interface and device policy.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <string_view>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/execution/execution_backend.hpp>
#include <SushiAI/graph/fusion.hpp>

namespace SushiBLAS
{
    namespace Device
    {
        /** @brief Declares the device profile, which the selector source reads. */
        struct Profile;
    } // namespace Device
} // namespace SushiBLAS

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
            /** @brief The form the static pass matched; never @c FusedForm::None. */
            FusedForm matched = FusedForm::None;

            /** @brief The element type every operand shares. */
            DType dtype = DType::Float32;

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
             * shorter entry to take part of it, or @c FusedForm::None to spell
             * the node out of unfused operations.
             */
            FusedForm form = FusedForm::None;

            /** @brief A sentence for the report; a static string, never owned. */
            std::string_view reason{};
        };

        /**
         * @class FusionSelector
         * @brief Defines the interface for bind-time selection of fused forms.
         */
        class FusionSelector
        {
            public:
                virtual ~FusionSelector() = default;

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
        class DeviceFusionSelector final : public FusionSelector
        {
            public:
                /**
                 * @brief Constructs a selector with explicit profile and execution mode.
                 * @param profile The device profile.
                 * @param mode The engine execution mode.
                 */
                DeviceFusionSelector(const SushiBLAS::Device::Profile& profile,
                                     SushiBLAS::ExecutionMode mode) noexcept;

                /**
                 * @brief Constructs a selector with explicit double-precision support and mode.
                 * @param supports_fp64 Whether the device has native double precision.
                 * @param mode The engine execution mode.
                 */
                DeviceFusionSelector(bool supports_fp64, SushiBLAS::ExecutionMode mode) noexcept;

                /**
                 * @brief Builds a selector describing the device and the mode of @p backend.
                 * @param backend Names the device through supports_fp64() and the mode through
                 *                its engine.
                 * @return A selector over those two facts.
                 */
                static DeviceFusionSelector for_backend(const ExecutionBackend& backend) noexcept;

                /** @brief Whether the device this selector reasons about has native double precision. */
                bool supports_fp64() const noexcept { return supports_fp64_; }

                /** @brief The execution mode this selector is reasoning about. */
                SushiBLAS::ExecutionMode mode() const noexcept { return mode_; }

                FusionChoice choose(const FusionQuery& query) const override;

            private:
                /** @brief Whether the device has native double precision. */
                bool supports_fp64_;
                /** @brief The execution mode the selector reasons about. */
                SushiBLAS::ExecutionMode mode_;
        };

        /**
         * @struct FusionSelection
         * @brief One selector decision, kept so the lowering can be interrogated.
         */
        struct FusionSelection
        {
            /** @brief What the static pass matched. */
            FusedForm matched = FusedForm::None;

            /** @brief What the selector chose. */
            FusedForm chosen = FusedForm::None;

            /** @brief Why. */
            std::string_view reason{};

            /** @brief How many device tasks the chosen spelling recorded. */
            std::size_t tasks = 0;
        };

    } // namespace Graph
} // namespace SushiAI
