/**************************************************************************/
/* model_records.cpp                                                      */
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
 * @file model_records.cpp
 * @brief Implements the writer and the strict reader of the model-state records.
 * @author Mustafa Garip
 */

#include "model_records.hpp"

#include <cstddef>
#include <string>
#include <unordered_set>
#include <vector>
#include <SushiBLAS/tensor/storage.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/allocation/host_copy.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/shape/tensor_shape.hpp>
#include <SushiAI/io/checkpoint.hpp>

namespace SushiAI::IO::Detail
{
    namespace
    {
        /** @brief Returns a zeroed host buffer as large as the elements of @p tensor. */
        [[nodiscard]] Result<std::vector<unsigned char>> host_buffer(
            const SushiBLAS::Tensor& tensor)
        {
            return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&tensor]()
            {
                return std::vector<unsigned char>(static_cast<std::size_t>(tensor.num_elements)
                                                  * dtype_size(tensor.dtype));
            });
        }
    } // namespace

    Status write_tensor_record(CheckpointWriter& writer, const std::string& name,
                               const SushiBLAS::Tensor& tensor)
    {
        SA_TRY_ASSIGN(std::vector<unsigned char> bytes, host_buffer(tensor));
        // Work submitted to the tensor's queue ends before its bytes are read.
        if (tensor.storage)
        {
            SA_TRY(SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Execute, [&tensor]()
            {
                tensor.storage->queue().wait();
                return Status();
            }));
        }
        SA_TRY(copy_to_host(tensor, 0, tensor.num_elements, dtype_size(tensor.dtype),
                            bytes.data()));

        return writer.write(record_for(name, tensor), bytes.data());
    }

    Status read_tensor_payload(CheckpointReader& reader, const TensorRecord& record,
                               const SushiBLAS::Tensor& tensor)
    {
        SA_TRY_ASSIGN(std::vector<unsigned char> bytes, host_buffer(tensor));
        SA_TRY(reader.read_payload(record, bytes.data()));

        return copy_from_host(tensor, 0, tensor.num_elements, dtype_size(tensor.dtype),
                              bytes.data());
    }

    Status write_model_records(CheckpointWriter& writer, NN::ModelState& state)
    {
        for (const NN::StateEntry& entry : state.entries())
        {
            SA_TRY_ASSIGN(SushiBLAS::Tensor* tensor, state.tensor(entry.state));
            SA_TRY(write_tensor_record(writer, entry.name, *tensor));
        }

        return Status();
    }

    Status read_model_records(CheckpointReader& reader, const std::filesystem::path& path,
                              NN::ModelState& state, SushiRuntime::span<const RenameRule> rules)
    {
        // The renamed names the file has supplied; a name in here is never accepted twice.
        std::unordered_set<std::string> supplied;
        while (reader.read_count() < reader.count())
        {
            SA_TRY_ASSIGN(const TensorRecord record, reader.next());

            SA_TRY_ASSIGN(const std::string target,
                          SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate,
                                                       [&]()
            {
                return renamed(record.name, rules);
            }));

            const Result<SushiBLAS::Tensor*> found = state.tensor(target);
            SA_FAIL_IF(!found.has_value(), StatusCode::Incompatible,
                       "'{}' holds '{}', which renames to '{}', and this model has no such "
                       "entry",
                       path.string(), record.name, target);
            SushiBLAS::Tensor* tensor = found.value();

            SA_TRY(SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate, [&]() -> Status
            {
                SA_FAIL_IF(!supplied.insert(target).second, StatusCode::Incompatible,
                           "'{}' supplies '{}' twice", path.string(), target);
                return Status();
            }));

            SA_FAIL_IF(record.dtype != tensor->dtype, StatusCode::Incompatible,
                       "'{}' holds '{}' as {}, but this model's is {}",
                       path.string(), record.name,
                       dtype_name(record.dtype), dtype_name(tensor->dtype));

            const Shape expected = shape_of(*tensor);
            SA_FAIL_IF(record.shape != expected, StatusCode::Incompatible,
                       "'{}' holds '{}' with shape {}, but this model's is {}",
                       path.string(), record.name,
                       record.shape.to_string(), expected.to_string());

            SA_TRY(read_tensor_payload(reader, record, *tensor));
        }

        for (const NN::StateEntry& entry : state.entries())
        {
            SA_FAIL_IF(supplied.count(entry.name) == 0, StatusCode::Incompatible,
                       "'{}' does not supply this model's '{}'", path.string(), entry.name);
        }

        return Status();
    }

} // namespace SushiAI::IO::Detail
