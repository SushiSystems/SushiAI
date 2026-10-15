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
 * @file fusion_selector.hpp
 * @brief Declares the dynamic fusion selector interface and device policy.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <string_view>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
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
