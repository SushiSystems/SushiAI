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

#include <algorithm>
#include <array>
#include <utility>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/allocation.hpp>
#include <SushiAI/core/seam.hpp>

namespace SushiAI
{
    namespace Data
    {
        MaterialisedDataset::MaterialisedDataset(std::vector<Batch> batches, int64_t batch_size,
                                                 const Shape& sample, int64_t classes,
                                                 std::string name)
            : batches_(std::move(batches)), batch_size_(batch_size), sample_shape_(sample),
              classes_(classes), name_(std::move(name))
        {

        }

        const Batch& MaterialisedDataset::batch(std::size_t index) const
        {
            SA_FATAL_IF(index >= batches_.size(),
                        "Batch {} is out of range; the dataset has {}", index, batches_.size());
            return batches_[index];
        }

        Result<Batch> allocate_batch(SushiBLAS::Engine& engine, int64_t rows,
                                     const Shape& sample, int64_t classes)
        {
            SA_FAIL_IF(sample.rank() + 1 > K_MAX_RANK, StatusCode::InvalidDescription,
                       "A batch of samples shaped {} has rank {}, above the maximum of {}",
                       sample.to_string(), sample.rank() + 1, K_MAX_RANK);

            std::array<int64_t, static_cast<std::size_t>(K_MAX_RANK)> input_dims{};
            input_dims[0] = rows;
            for (int32_t d = 0; d < sample.rank(); ++d)
                input_dims[static_cast<std::size_t>(d) + 1] = sample[d];

            const int64_t target_dims[2] = {rows, classes};

            Batch batch;
            SA_TRY_ASSIGN(batch.inputs,
                          allocate_tensor(engine,
                                          SushiRuntime::span<const int64_t>(
                                              input_dims.data(),
                                              static_cast<std::size_t>(sample.rank()) + 1)));
            SA_TRY_ASSIGN(batch.targets,
                          allocate_tensor(engine,
                                          SushiRuntime::span<const int64_t>(target_dims, 2)));
            SA_TRY(Detail::guard(Detail::SeamPhase::Allocate, [&]()
            {
                batch.labels.resize(static_cast<std::size_t>(rows));
                return Status();
            }));

            return std::move(batch);
        }

        void write_one_hot(SushiBLAS::Tensor& targets, const std::vector<int32_t>& labels,
                           int64_t classes)
        {
            float* data = targets.data_as<float>();
            std::fill(data, data + targets.num_elements, 0.0f);

            for (std::size_t row = 0; row < labels.size(); ++row)
                data[static_cast<int64_t>(row) * classes + labels[row]] = 1.0f;
        }

    } // namespace Data
} // namespace SushiAI
