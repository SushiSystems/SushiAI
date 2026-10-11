/**************************************************************************/
/* table_position_feed.hpp                                                */
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
 * @file table_position_feed.hpp
 * @brief Declares TablePositionFeed, which places a token by its row of a position table.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/train/evaluation_step.hpp>
#include "position_feed.hpp"

namespace SushiAI::Language::Generation::Detail
{
    /**
     * @brief Feeds one [1, 1] int32 leaf with the position of the token in the window.
     *
     * A slide gives every kept token another row of the table, so no cached row stays right.
     */
    class TablePositionFeed final : public PositionFeed
    {
        public:
            /**
             * @brief Allocates the feed of the leaf @p name of @p step.
             * @return The feed, or InvalidDescription for a leaf that is not [1, 1] int32.
             */
            [[nodiscard]] static Result<std::unique_ptr<PositionFeed>> make(
                SushiBLAS::Engine& engine, const Train::EvaluationStep& step,
                std::string_view name);

            TablePositionFeed(const TablePositionFeed&) = delete;
            TablePositionFeed& operator=(const TablePositionFeed&) = delete;
            TablePositionFeed(TablePositionFeed&&) = delete;
            TablePositionFeed& operator=(TablePositionFeed&&) = delete;
            ~TablePositionFeed() override = default;

            /** @brief Writes @p position, which is below T while every slide resets the step. */
            void place(std::int64_t position) noexcept override;

            /** @brief Lists the one leaf. */
            [[nodiscard]] SushiRuntime::span<const Train::NamedFeed> feeds()
                const noexcept override;

            /** @brief Returns false. */
            [[nodiscard]] bool keeps_rows_on_slide() const noexcept override;

        private:
            /** @brief Takes the tensor make() allocated and the name of its leaf. */
            TablePositionFeed(Data::Batch position, std::string_view name);

            /// @brief Holds the [1, 1] int32 position a run feeds.
            Data::Batch position_;

            /// @brief Names the leaf of the step.
            std::string name_;

            /// @brief Pairs name_ with position_; it points into this object.
            Train::NamedFeed feed_;
    };

} // namespace SushiAI::Language::Generation::Detail
