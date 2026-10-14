/**************************************************************************/
/* ring_position_feed.cpp                                                 */
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
 * @file ring_position_feed.cpp
 * @brief Implements RingPositionFeed.
 * @author Mustafa Garip
 */

#include <SushiAI/language/generation/ring_position_feed.hpp>

#include <utility>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/core/check.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/seam.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/data/materialised_dataset.hpp>
#include "step_leaf.hpp"

namespace SushiAI::Language::Generation::Detail
{
    Result<std::unique_ptr<PositionFeed>> RingPositionFeed::make(
        SushiBLAS::Engine& engine, const Train::EvaluationStep& step,
        const RingStepLeaves& leaves, std::int64_t window)
    {
        SA_FAIL_IF(leaves.angles == nullptr, StatusCode::InvalidDescription,
                   "{}: the step is fed the cache slot '{}', but the ring names no function "
                   "that fills '{}' and '{}'",
                   K_CACHED_GENERATOR, leaves.slot, leaves.cosine, leaves.sine);

        SA_TRY(fitting_leaf(step, leaves.slot, 1, DType::Int32));
        SA_TRY_ASSIGN(const SushiBLAS::Tensor* const cosine,
                      fitting_leaf(step, leaves.cosine, K_ANY_COLUMNS, DType::Float32));
        SA_TRY(fitting_leaf(step, leaves.sine, cosine->shape[1], DType::Float32));

        Feeds feeds;
        const Data::InputDescription angles{Shape{cosine->shape[1]}, DType::Float32};
        SA_TRY(Data::prepare_batch(engine, 1, Data::InputDescription{Shape{1}, DType::Int32},
                                   {}, feeds.slot));
        SA_TRY(Data::prepare_batch(engine, 1, angles, {}, feeds.cosine));
        SA_TRY(Data::prepare_batch(engine, 1, angles, {}, feeds.sine));

        return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
        {
            return std::unique_ptr<PositionFeed>(
                new RingPositionFeed(std::move(feeds), leaves, window));
        });
    }

    RingPositionFeed::RingPositionFeed(Feeds feeds, const RingStepLeaves& leaves,
                                       std::int64_t window)
        : feeds_(std::move(feeds)),
          names_{std::string(leaves.slot), std::string(leaves.cosine),
                 std::string(leaves.sine)},
          named_{{names_[0], &feeds_.slot.inputs},
                 {names_[1], &feeds_.cosine.inputs},
                 {names_[2], &feeds_.sine.inputs}},
          angles_(leaves.angles),
          window_(window)
    {

    }

    void RingPositionFeed::place(std::int64_t position) noexcept
    {
        feeds_.slot.inputs.data_as<std::int32_t>()[0] =
            static_cast<std::int32_t>(position % window_);

        const std::size_t pairs = static_cast<std::size_t>(feeds_.cosine.inputs.shape[1]);
        angles_(position,
                SushiRuntime::span<float>(feeds_.cosine.inputs.data_as<float>(), pairs),
                SushiRuntime::span<float>(feeds_.sine.inputs.data_as<float>(), pairs));
    }

    SushiRuntime::span<const Train::NamedFeed> RingPositionFeed::feeds() const noexcept
    {
        return SushiRuntime::span<const Train::NamedFeed>(named_, K_LEAVES);
    }

    bool RingPositionFeed::keeps_rows_on_slide() const noexcept
    {
        return true;
    }

} // namespace SushiAI::Language::Generation::Detail
