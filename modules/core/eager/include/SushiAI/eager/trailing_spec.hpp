/**************************************************************************/
/* trailing_spec.hpp                                                      */
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
 * @file trailing_spec.hpp
 * @brief Declares TrailingSpec, which names the work a step records behind its regions.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#pragma once

#include <vector>

#include <SushiAI/eager/storage_id.hpp>
#include <SushiAI/graph/step_recorder.hpp>

namespace SushiAI
{
    namespace Eager
    {
        /** @brief Names one gradient tensor that is copied into the tensor of another state. */
        struct TrailingCopy
        {
            StorageID from = K_NO_STORAGE; /**< The tensor the regions write. */
            StorageID to = K_NO_STORAGE;   /**< The tensor that receives the same bytes. */
        };

        /** @brief Names one running statistic and the step value blended into it. */
        struct TrailingBlend
        {
            StorageID target = K_NO_STORAGE; /**< The state tensor the blend writes. */
            StorageID source = K_NO_STORAGE; /**< The tensor the step left the value in. */
            double momentum = 0.0;           /**< The weight of the source. */
            double source_scale = 1.0;       /**< The factor the source is scaled by. */
        };

        /** @brief Returns whether @p a and @p b name the same copy. */
        [[nodiscard]] inline bool operator==(const TrailingCopy& a, const TrailingCopy& b) noexcept
        {
            return a.from == b.from && a.to == b.to;
        }

        /** @brief Returns whether @p a and @p b name the same blend. */
        [[nodiscard]] inline bool operator==(const TrailingBlend& a,
                                             const TrailingBlend& b) noexcept
        {
            return a.target == b.target && a.source == b.source && a.momentum == b.momentum
                   && a.source_scale == b.source_scale;
        }

        /**
         * @brief Holds the work behind a step's regions as storage ids, so two steps compare.
         *
         * The work runs in this order: the copies, the caller's recorder, the blends.
         */
        struct TrailingSpec
        {
            /** @brief The caller's recorder, or null; not owned. */
            Graph::StepRecorder* inner = nullptr;

            /** @brief The gradient copies, in the order they run. */
            std::vector<TrailingCopy> copies;

            /** @brief The running-statistic blends, in the order they run. */
            std::vector<TrailingBlend> blends;

            /** @brief Returns whether the caller's recorder is all the step records. */
            [[nodiscard]] bool is_plain() const noexcept
            {
                return copies.empty() && blends.empty();
            }
        };

        /** @brief Returns whether @p a and @p b record the same work in the same order. */
        [[nodiscard]] inline bool operator==(const TrailingSpec& a, const TrailingSpec& b) noexcept
        {
            return a.inner == b.inner && a.copies == b.copies && a.blends == b.blends;
        }

    } // namespace Eager
} // namespace SushiAI
