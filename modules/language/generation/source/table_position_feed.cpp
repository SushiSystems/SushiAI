/**************************************************************************/
/* table_position_feed.cpp                                                */
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
 * @file table_position_feed.cpp
 * @brief Implements TablePositionFeed.
 * @author Mustafa Garip
 */

#include "table_position_feed.hpp"

#include <utility>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/seam.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/data/materialised_dataset.hpp>
#include "step_leaf.hpp"

namespace SushiAI::Language::Generation::Detail
{
    Result<std::unique_ptr<PositionFeed>> TablePositionFeed::make(
        SushiBLAS::Engine& engine, const Train::EvaluationStep& step, std::string_view name)
    {
        SA_TRY(fitting_leaf(step, name, 1, DType::Int32));

        Data::Batch position;
        SA_TRY(Data::prepare_batch(engine, 1, Data::InputDescription{Shape{1}, DType::Int32},
                                   {}, position));

        return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
        {
            return std::unique_ptr<PositionFeed>(
                new TablePositionFeed(std::move(position), name));
        });
    }

    TablePositionFeed::TablePositionFeed(Data::Batch position, std::string_view name)
        : position_(std::move(position)), name_(name), feed_{name_, &position_.inputs}
    {

    }

    void TablePositionFeed::place(std::int64_t position) noexcept
    {
        position_.inputs.data_as<std::int32_t>()[0] = static_cast<std::int32_t>(position);
    }

    SushiRuntime::span<const Train::NamedFeed> TablePositionFeed::feeds() const noexcept
    {
        return SushiRuntime::span<const Train::NamedFeed>(&feed_, 1);
    }

    bool TablePositionFeed::keeps_rows_on_slide() const noexcept
    {
        return false;
    }

} // namespace SushiAI::Language::Generation::Detail
