/**************************************************************************/
/* extra_input_batches.hpp                                                */
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
 * @file extra_input_batches.hpp
 * @brief Declares the batch of the extra-input sample model that the tests of the trainer and of the copy build.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/sample_models/extra_input_model.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>

namespace SushiAI::Test
{
    /** @brief Counts the samples of every batch of the extra-input tests. */
    inline constexpr std::int64_t K_EXTRA_BATCH = 4;

    /** @brief Returns the extents of the input of one batch: [K_EXTRA_BATCH, 2, 2, 3]. */
    [[nodiscard]] inline Shape extra_batch_extents()
    {
        return Shape{K_EXTRA_BATCH, 2, 2, 3};
    }

    /** @brief Returns @p count values of eighths between -1 and 1, stepping @p stride a value. */
    [[nodiscard]] inline std::vector<double> eighths(std::int64_t count, std::int64_t stride)
    {
        std::vector<double> values;
        for (std::int64_t index = 0; index < count; ++index)
            values.push_back(static_cast<double>((index * stride) % 17 - 8) / 8.0);

        return values;
    }

    /**
     * @brief Returns a batch of @p inputs, the labels 0 1 2 1 and the role "offset" of @p offsets.
     * @param offset_extents The shape the role's tensor is allocated with.
     */
    [[nodiscard]] inline Result<Data::Batch> make_extra_input_batch(
        SushiBLAS::Engine& engine, const std::vector<double>& inputs,
        const std::vector<double>& offsets, const Shape& offset_extents)
    {
        Data::Batch batch;
        SA_TRY_ASSIGN(batch.inputs, allocate_tensor(engine, extra_batch_extents(), DType::Float32,
                                                    TensorAccess::Host));
        write_tensor(batch.inputs, inputs);

        Data::BatchTarget labels;
        labels.role = std::string(Data::K_ROLE_LABELS);
        SA_TRY_ASSIGN(labels.values, allocate_tensor(engine, Shape{K_EXTRA_BATCH}, DType::Int32,
                                                     TensorAccess::Host));
        write_tensor(labels.values, std::vector<double>{0.0, 1.0, 2.0, 1.0});
        batch.targets.push_back(std::move(labels));

        Data::BatchTarget offset;
        offset.role = std::string(K_EXTRA_INPUT_OFFSET);
        SA_TRY_ASSIGN(offset.values, allocate_tensor(engine, offset_extents, DType::Float32,
                                                     TensorAccess::Host));
        write_tensor(offset.values, offsets);
        batch.targets.push_back(std::move(offset));

        return batch;
    }

    /** @brief Returns the batch of make_extra_input_batch() whose offset has the shape of the input. */
    [[nodiscard]] inline Result<Data::Batch> make_extra_input_batch(
        SushiBLAS::Engine& engine, const std::vector<double>& inputs,
        const std::vector<double>& offsets)
    {
        return make_extra_input_batch(engine, inputs, offsets, extra_batch_extents());
    }

} // namespace SushiAI::Test
