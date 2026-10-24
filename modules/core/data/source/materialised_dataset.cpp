/**************************************************************************/
/* materialised_dataset.cpp                                               */
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
 * @file materialised_dataset.cpp
 * @brief Implements the dataset that holds device batches, and the helpers that fill one.
 * @author Mustafa Garip
 */

#include <SushiAI/data/materialised_dataset.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/data/target.hpp>

namespace SushiAI
{
    namespace Data
    {
        namespace
        {
            /**
             * @brief Allocates a [rows, sample...] tensor of @p dtype.
             * @return The tensor, InvalidDescription when the shape exceeds the maximum rank,
             *         or the failure of the allocation.
             */
            [[nodiscard]] Result<SushiBLAS::Tensor> allocate_rows(SushiBLAS::Engine& engine,
                                                                  int64_t rows,
                                                                  const Shape& sample,
                                                                  DType dtype)
            {
                SA_FAIL_IF(sample.rank() + 1 > K_MAX_RANK, StatusCode::InvalidDescription,
                           "A batch of samples shaped {} has rank {}, above the maximum of {}",
                           sample.to_string(), sample.rank() + 1, K_MAX_RANK);

                std::array<int64_t, static_cast<std::size_t>(K_MAX_RANK)> dims{};
                dims[0] = rows;
                for (int32_t d = 0; d < sample.rank(); ++d)
                    dims[static_cast<std::size_t>(d) + 1] = sample[d];

                return allocate_tensor(engine,
                                       SushiRuntime::span<const int64_t>(
                                           dims.data(),
                                           static_cast<std::size_t>(sample.rank()) + 1),
                                       dtype);
            }

            /** @brief Reports whether @p tensor is [rows, sample...] of @p dtype, held alone. */
            bool holds_alone(const SushiBLAS::Tensor& tensor, int64_t rows, const Shape& sample,
                             DType dtype) noexcept
            {
                if (!tensor.storage || tensor.storage->ref_count() != 1)
                    return false;
                if (tensor.storage_offset != 0 || tensor.dtype != dtype
                    || tensor.layout != SushiBLAS::Core::Layout::RowMajor
                    || tensor.rank != sample.rank() + 1 || tensor.shape[0] != rows)
                    return false;

                for (int32_t d = 0; d < sample.rank(); ++d)
                {
                    if (tensor.shape[static_cast<std::size_t>(d) + 1] != sample[d])
                        return false;
                }
                return true;
            }

            /** @brief Reports whether @p batch holds, alone, the layout prepare_batch() names. */
            bool holds_layout(const Batch& batch, int64_t rows, const InputDescription& input,
                              const std::vector<TargetDescription>& targets) noexcept
            {
                if (batch.targets.size() != targets.size()
                    || !holds_alone(batch.inputs, rows, input.sample_shape, input.dtype))
                    return false;

                for (std::size_t t = 0; t < targets.size(); ++t)
                {
                    if (batch.targets[t].role != targets[t].role
                        || !holds_alone(batch.targets[t].values, rows, targets[t].sample_shape,
                                        targets[t].dtype))
                        return false;
                }
                return true;
            }
        } // namespace

        MaterialisedDataset::MaterialisedDataset(std::vector<Batch> batches, int64_t batch_size,
                                                 const InputDescription& input,
                                                 std::vector<TargetDescription> targets,
                                                 std::string name)
            : batches_(std::move(batches)), batch_size_(batch_size), input_(input),
              targets_(std::move(targets)), name_(std::move(name))
        {

        }

        Status MaterialisedDataset::read(std::size_t index, Batch& batch) const
        {
            SA_FATAL_IF(index >= batches_.size(),
                        "Batch {} is out of range; the dataset has {}", index, batches_.size());
            return share_batch(batches_[index], batch);
        }

        Status MaterialisedDataset::read_rows(std::size_t index, RowRange rows,
                                              Batch& batch) const
        {
            check_rows(*this, index, rows);
            return share_rows(batches_[index], rows, batch);
        }

        Result<Batch> allocate_batch(SushiBLAS::Engine& engine, int64_t rows,
                                     const InputDescription& input,
                                     const std::vector<TargetDescription>& targets)
        {
            Batch batch;
            SA_TRY_ASSIGN(batch.inputs,
                          allocate_rows(engine, rows, input.sample_shape, input.dtype));
            SA_TRY(Detail::guard(Detail::SeamPhase::Allocate, [&]()
            {
                batch.targets.reserve(targets.size());
                return Status();
            }));

            for (const TargetDescription& target : targets)
            {
                SA_TRY_ASSIGN(SushiBLAS::Tensor values,
                              allocate_rows(engine, rows, target.sample_shape, target.dtype));
                SA_TRY(Detail::guard(Detail::SeamPhase::Allocate, [&]()
                {
                    batch.targets.push_back(BatchTarget{target.role, std::move(values)});
                    return Status();
                }));
            }

            return std::move(batch);
        }

        Status share_batch(const Batch& held, Batch& batch) noexcept
        {
            return Detail::guard(Detail::SeamPhase::Allocate, [&]()
            {
                batch = held;
                return Status();
            });
        }

        Status share_rows(const Batch& held, RowRange rows, Batch& batch) noexcept
        {
            const int64_t end = rows.first + rows.count;
            return Detail::guard(Detail::SeamPhase::Allocate, [&]()
            {
                batch.inputs = held.inputs.slice(0, rows.first, end);
                batch.targets.resize(held.targets.size());
                for (std::size_t t = 0; t < held.targets.size(); ++t)
                {
                    batch.targets[t].role = held.targets[t].role;
                    batch.targets[t].values = held.targets[t].values.slice(0, rows.first, end);
                }
                return Status();
            });
        }

        Status prepare_batch(SushiBLAS::Engine& engine, int64_t rows,
                             const InputDescription& input,
                             const std::vector<TargetDescription>& targets, Batch& batch)
        {
            if (holds_layout(batch, rows, input, targets))
                return Status();

            SA_TRY_ASSIGN(batch, allocate_batch(engine, rows, input, targets));
            return Status();
        }

    } // namespace Data
} // namespace SushiAI
