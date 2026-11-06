/**************************************************************************/
/* best_shape_head_delta_rule.cpp                                         */
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
 * @file best_shape_head_delta_rule.cpp
 * @brief Implements BestShapeHeadDeltaRule over compute_head_delta and measure_head_cost.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/training/best_shape_head_delta_rule.hpp>

#include <cstddef>
#include <cstdint>

#include <SushiAI/failure/check.hpp>

namespace SushiAI::Vision::Training
{
    Status BestShapeHeadDeltaRule::check(const Detect::Anchor::AnchorLayout&,
                                       SushiRuntime::span<const AnchorHead> heads) const
    {
        for (const AnchorHead& head : heads)
        {
            SA_FAIL_IF(head.options.size != Detect::Anchor::SizeForm::Exponential,
                       StatusCode::InvalidDescription,
                       "the best_shape assignment decodes sides as exponentials; head '{}' "
                       "states another 'size'",
                       head.output);
        }

        return Status();
    }

    Result<Detect::Anchor::HeadCost> BestShapeHeadDeltaRule::compute(
        const HeadDeltaBatch& batch, SushiRuntime::span<Detect::Anchor::HeadDelta> deltas) const
    {
        SA_FATAL_IF(batch.outputs.size() != batch.heads.size()
                        || deltas.size() != batch.heads.size(),
                    "the best-shape rule was given {} outputs and {} deltas for {} heads",
                    batch.outputs.size(), deltas.size(), batch.heads.size());

        for (std::size_t head = 0; head < batch.heads.size(); ++head)
        {
            const AnchorHead& described = batch.heads[head];
            SA_TRY(Detect::Anchor::compute_head_delta(
                batch.outputs[head], described.shape, described.options, *batch.layout,
                static_cast<std::int64_t>(head), described.training, *batch.truth,
                batch.network, deltas[head]));
        }

        Detect::Anchor::HeadCost sum;
        for (std::size_t head = 0; head < batch.heads.size(); ++head)
        {
            const AnchorHead& described = batch.heads[head];
            const Detect::Anchor::HeadCost cost = Detect::Anchor::measure_head_cost(
                deltas[head], described.shape, described.options, described.training);

            sum.total += cost.total;
            sum.box += cost.box;
            sum.non_box += cost.non_box;
            sum.stored += cost.stored;
        }

        return sum;
    }

} // namespace SushiAI::Vision::Training
