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
 * @brief Declares the interface that turns the fitted samples of a batch into a task's roles.
 * @author Mustafa Garip
 */

#pragma once

#include <string_view>
#include <vector>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/vision/datasets/fitted_sample.hpp>
#include <SushiAI/vision/datasets/sample.hpp>

/** @brief Holds the readers of the on-disk layouts vision datasets are kept in. */
namespace SushiAI::Vision::Datasets
{
    /**
     * @brief Writes the inputs and the roles of one task from the samples of one batch.
     * @see docs/design/DATASETS.md, section "The parts of a dataset".
     */
    class Collator
    {
        public:
            /** @brief Destroys the collator through the interface. */
            virtual ~Collator() = default;

            /** @brief Returns the name of the task, as a training file spells it. */
            [[nodiscard]] virtual std::string_view task() const noexcept = 0;

            /** @brief Returns the roles collate() writes, each with its per-sample shape. */
            [[nodiscard]] virtual const std::vector<Data::TargetDescription>& targets()
                const noexcept = 0;

            /**
             * @brief Writes Batch::inputs and every role of the task.
             * @param list The list the samples came from; read for file names.
             * @param samples One per row of the batch, each at the input size.
             * @param batch Allocated by Data::allocate_batch over targets().
             * @return Incompatible for a batch of another shape; InvalidDescription, opening
             *         with the file, for a sample the task cannot use.
             */
            [[nodiscard]] virtual Status collate(const SampleList& list,
                                                 SushiRuntime::span<const FittedSample> samples,
                                                 Data::Batch& batch) const = 0;
    };
} // namespace SushiAI::Vision::Datasets
