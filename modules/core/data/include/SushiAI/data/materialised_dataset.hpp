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
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/data/target.hpp>

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
                 *                samples as @p input describes them and one target per entry
                 *                of @p targets, in that order.
                 * @param name The name a run report prints.
                 */
                MaterialisedDataset(std::vector<Batch> batches, int64_t batch_size,
                                    const InputDescription& input,
                                    std::vector<TargetDescription> targets, std::string name) noexcept;

                int64_t batch_size() const noexcept override { return batch_size_; }
                Shape sample_shape() const override { return input_.sample_shape; }
                DType sample_dtype() const noexcept override { return input_.dtype; }
                const std::vector<TargetDescription>& targets() const noexcept override
                {
                    return targets_;
                }
                std::size_t batch_count() const noexcept override { return batches_.size(); }
                [[nodiscard]] Status read(std::size_t index, Batch& batch) const override;
                [[nodiscard]] Status read_rows(std::size_t index, RowRange rows,
                                               Batch& batch) const override;
                std::string_view name() const noexcept override { return name_; }

            private:
                /** @brief Holds every batch of the epoch, in order. */
                std::vector<Batch> batches_;

                /** @brief Holds the samples per batch. */
                int64_t batch_size_;

                /** @brief Holds the per-sample shape and the element type of the inputs. */
                InputDescription input_;

                /** @brief Holds the description of every role the batches carry. */
                std::vector<TargetDescription> targets_;

                /** @brief Holds the name a run report prints. */
                std::string name_;
        };

        /**
         * @brief Allocates the input tensor of one batch and one tensor per target.
         * @param input The input tensor is [rows, sample_shape...] of the type it names.
         * @param targets One description per role; its tensor is [rows, sample_shape...].
         * @return The batch, its targets in the order of @p targets and its contents unset;
         *         InvalidDescription when a shape exceeds the maximum rank, or the failure of
         *         an allocation.
         */
        [[nodiscard]] Result<Batch> allocate_batch(SushiBLAS::Engine& engine, int64_t rows,
                                                   const InputDescription& input,
                                                   const std::vector<TargetDescription>& targets) noexcept;

        /**
         * @brief Makes @p batch share the tensors of @p held; no element is copied.
         * @param batch Overwritten; its role list is reused when it is long enough.
         * @return Success, or OutOfMemory when the role list of @p batch cannot grow.
         */
        [[nodiscard]] Status share_batch(const Batch& held, Batch& batch) noexcept;

        /**
         * @brief Makes @p batch view rows @p rows of every tensor of @p held; no element is
         *        copied.
         * @param held Its inputs and every target hold one row per sample on axis 0.
         * @param batch Overwritten; its role list is reused when it is long enough.
         * @return Success, or OutOfMemory when the role list of @p batch cannot grow.
         * @pre @p rows lies inside axis 0 of every tensor of @p held.
         */
        [[nodiscard]] Status share_rows(const Batch& held, RowRange rows, Batch& batch) noexcept;

        /**
         * @brief Leaves @p batch with tensors of one layout that nothing else references.
         * @param batch Kept as it is when it already holds that layout alone; replaced by
         *              allocate_batch() otherwise, so storage a dataset shares is not written.
         * @return Success with the contents unset, or the failure of allocate_batch().
         */
        [[nodiscard]] Status prepare_batch(SushiBLAS::Engine& engine, int64_t rows,
                                           const InputDescription& input,
                                           const std::vector<TargetDescription>& targets,
                                           Batch& batch) noexcept;

    } // namespace Data
} // namespace SushiAI
