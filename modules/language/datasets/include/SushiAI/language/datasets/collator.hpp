/**************************************************************************/
/* collator.hpp                                                           */
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
 * @file collator.hpp
 * @brief Declares the interface that turns the token windows of a batch into a task's roles.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/language/text/token.hpp>

/** @brief Holds the token file of a corpus and the dataset that reads it in windows. */
namespace SushiAI::Language::Datasets
{
    /**
     * @brief Writes the inputs and the roles of one task from the windows of one batch.
     * @see docs/design/DATASETS/README.md, section "The parts of a dataset".
     */
    class Collator
    {
        public:
            /** @brief Destroys the collator through the interface. */
            virtual ~Collator() = default;

            /** @brief Returns the name of the task, as a training file spells it. */
            [[nodiscard]] virtual std::string_view task() const noexcept = 0;

            /** @brief Returns T, the positions of one input row. */
            [[nodiscard]] virtual std::int64_t window_length() const noexcept = 0;

            /** @brief Returns how many ids of the file the task reads for one row. */
            [[nodiscard]] virtual std::int64_t tokens_per_window() const noexcept = 0;

            /** @brief Returns the token count every id it is given must be below. */
            [[nodiscard]] virtual std::int32_t vocabulary_size() const noexcept = 0;

            /** @brief Returns the roles collate() writes, each with its per-sample shape. */
            [[nodiscard]] virtual const std::vector<Data::TargetDescription>& targets()
                const noexcept = 0;

            /**
             * @brief Writes Batch::inputs, int32 of [N, T], and every role of the task.
             * @param tokens N windows of tokens_per_window() ids each, one after another,
             *               every id in [0, vocabulary_size()).
             * @param batch Holds an int32 input of [N, T] and one tensor per entry of targets().
             * @return Success, or Incompatible for a token count or a batch of another shape.
             */
            [[nodiscard]] virtual Status collate(SushiRuntime::span<const Text::TokenID> tokens,
                                                 Data::Batch& batch) const = 0;
    };
} // namespace SushiAI::Language::Datasets
