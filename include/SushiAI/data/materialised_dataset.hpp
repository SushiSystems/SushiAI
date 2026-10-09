/**************************************************************************/
/* materialised_dataset.hpp                                               */
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
 * @file materialised_dataset.hpp
 * @brief Declares the dataset that holds device batches, and the helpers that fill one.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <SushiBLAS/engine.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/data/dataset.hpp>

namespace SushiAI
{
    namespace Data
    {
        /**
         * @class MaterialisedDataset
         * @brief A dataset whose batches are already on the device.
         */
        class MaterialisedDataset final : public Dataset
        {
            public:
                /**
                 * @brief Holds @p batches and the extents they were built with.
                 * @param batches Every batch of the epoch, in order; each holds @p batch_size
                 *                samples shaped @p sample and targets @p classes wide.
                 * @param name The name a run report prints.
                 */
                MaterialisedDataset(std::vector<Batch> batches, int64_t batch_size,
                                    const Shape& sample, int64_t classes, std::string name);

                int64_t batch_size() const noexcept override { return batch_size_; }
                int64_t input_dim() const noexcept override
                {
                    return sample_shape_.num_elements();
                }
                Shape sample_shape() const override { return sample_shape_; }
                int64_t num_classes() const noexcept override { return classes_; }
                std::size_t batch_count() const noexcept override { return batches_.size(); }
                const Batch& batch(std::size_t index) const override;
                std::string_view name() const noexcept override { return name_; }

            private:
                /** @brief Holds every batch of the epoch, in order. */
                std::vector<Batch> batches_;

                /** @brief Holds the samples per batch. */
                int64_t batch_size_;

                /** @brief Holds the shape of one sample, batch axis left out. */
                Shape sample_shape_;

                /** @brief Holds the width of the target matrix. */
                int64_t classes_;

                /** @brief Holds the name a run report prints. */
                std::string name_;
        };

        /**
         * @brief Allocates the tensors of one batch and sizes its label list.
         * @param rows Samples in the batch.
         * @param sample The shape of one sample; the input tensor is [rows, sample...].
         * @param classes Width of the target matrix.
         * @return The batch, its contents unset; InvalidDescription when [rows, sample...]
         *         exceeds the maximum rank, or the failure of an allocation.
         */
        [[nodiscard]] Result<Batch> allocate_batch(SushiBLAS::Engine& engine, int64_t rows,
                                                   const Shape& sample, int64_t classes);

        /**
         * @brief Writes a one-hot row pattern straight into shared USM.
         * @param targets A [labels.size(), classes] float tensor.
         * @param labels One class index per row, each in [0, classes).
         */
        void write_one_hot(SushiBLAS::Tensor& targets, const std::vector<int32_t>& labels,
                           int64_t classes);

    } // namespace Data
} // namespace SushiAI
