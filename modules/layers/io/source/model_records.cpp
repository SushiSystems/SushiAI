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
 * @file model_records.cpp
 * @brief Implements the writer and the strict reader of the model-state records.
 * @author Mustafa Garip
 */

#include "model_records.hpp"

#include <string>
#include <unordered_set>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/shape/tensor_shape.hpp>
#include <SushiAI/io/checkpoint.hpp>

namespace SushiAI::IO::Detail
{
    Status write_model_records(CheckpointWriter& writer, NN::ModelState& state)
    {
        for (const NN::StateEntry& entry : state.entries())
        {
            SA_TRY_ASSIGN(SushiBLAS::Tensor* tensor, state.tensor(entry.state));
            SA_TRY(writer.write(record_for(entry.name, *tensor), tensor->data()));
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

            // Payloads are streamed directly into the state's own memory.
            SA_TRY(reader.read_payload(record, tensor->data()));
        }

        for (const NN::StateEntry& entry : state.entries())
        {
            SA_FAIL_IF(supplied.count(entry.name) == 0, StatusCode::Incompatible,
                       "'{}' does not supply this model's '{}'", path.string(), entry.name);
        }

        return Status();
    }

} // namespace SushiAI::IO::Detail
