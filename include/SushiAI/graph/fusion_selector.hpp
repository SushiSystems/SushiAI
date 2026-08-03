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

#pragma once

/**
 * @file fusion_selector.hpp
 * @brief The dynamic half of fusion: which matched form to actually spell.
 *
 * @c graph/fusion.hpp decides **what could be fused** - a question about the
 * graph, answerable with no device present, which is why that pass is in the
 * host lane and unit-tested without one. This file decides **which form to
 * use**, which is a question about the machine and cannot be answered until
 * bind time: the device's profile, the engine's execution mode and the operands'
 * actual layout are none of them known while a model is being traced.
 *
 * Keeping the two apart is the point. A device fact inside the static pass
 * would mean a graph that fuses differently on two machines, which makes the
 * IR - the thing every other pass reasons over, and the thing a test compares -
 * depend on where it was built. Here, the IR is the same everywhere and only
 * the spelling of one node changes.
 *
 * @section narrowing Declining is always expressible
 *
 * A selector may narrow a matched form to a shorter one, or refuse it outright,
 * and the lowering can always honour that: a @c GEMM_BIAS_RELU node holds
 * @c [A, B, bias] and writes one result, so "GEMM with a bias epilogue, then a
 * separate in-place ReLU" and "GEMM, then a rank-1 bias update, then a ReLU"
 * are both spellings of the same node with the same operands and no extra
 * buffer. Every spelling is bit-identical to every other - see
 * @c graph/fusion.hpp's note on @c -ffp-contract=off - so a decline costs
 * launches and changes no number.
 *
 * @section rules What the default selector actually decides on today
 *
 * Two rules, both correctness rules, and it is worth saying plainly that there
 * is no *performance* rule here yet. Every entry in today's catalog removes
 * whole passes over the result tensor at every problem size and on both of
 * SushiBLAS's GEMM variants - the local-tiled one and the no-local one apply
 * the epilogue at the same point - so no device fact argues against any of
 * them. Inventing a threshold that cannot be measured on the one device this
 * machine has would be worse than saying so. The seam exists because the next
 * catalog entry may well need it: a GELU epilogue spilling its pre-activation
 * to an auxiliary buffer is exactly a form whose admissibility depends on
 * whether that buffer fits.
 */

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
         * @brief The bind-time choice between catalog entries.
         *
         * An interface rather than a free function so a test can drive the
         * lowering with a device profile this machine does not have. There is
         * exactly one OpenCL CPU device here; a policy that can only be
         * exercised on hardware nobody owns is a policy nobody has run.
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
         * @brief The default policy: the device's profile and the engine's mode.
         *
         * Holds a @c Device::Profile **by value**. The profile is a small
         * aggregate of answers SushiBLAS caches per device for the life of the
         * process, so copying it costs nothing and removes any question about
         * the reference outliving the selector.
         */
        class DeviceFusionSelector final : public IFusionSelector
        {
            public:
                /**
                 * @brief Builds a selector over an explicit profile and mode.
                 *
                 * This is the constructor a test uses: it can describe a GPU
                 * with dedicated scratchpad memory, or a device with no double
                 * precision, on a machine that has neither.
                 *
                 * @param profile The device's answers.
                 * @param mode The engine's execution mode.
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
