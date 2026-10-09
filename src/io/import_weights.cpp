/**************************************************************************/
/* import_weights.cpp                                                     */
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
 * @file import_weights.cpp
 * @brief Defines the strict checkpoint loader and the prefix rules it applies to record names.
 * @author Mustafa Garip
 */

#include <SushiAI/io/import_weights.hpp>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <string>
#include <string_view>
#include <unordered_set>
#include <SushiBLAS/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/seam.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/tensor_shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/io/checkpoint_format.hpp>
#include <SushiAI/nn/model_state.hpp>

namespace SushiAI
{
    namespace IO
    {
        namespace
        {
            /**
             * @brief Returns whether @p name begins with @p prefix.
             * @param name The text to test.
             * @param prefix The text it must begin with; the empty prefix matches every name.
             * @return True when name starts with prefix.
             */
            bool starts_with(std::string_view name, std::string_view prefix) noexcept
            {
                return name.size() >= prefix.size() && name.substr(0, prefix.size()) == prefix;
            }
        } // namespace

        std::string renamed(std::string_view name, SushiRuntime::span<const RenameRule> rules)
        {
            for (const RenameRule& rule : rules)
            {
                if (!starts_with(name, rule.from_prefix))
                    continue;

                std::string out;
                out.reserve(rule.to_prefix.size() + name.size() - rule.from_prefix.size());
                out.append(rule.to_prefix);
                out.append(name.substr(rule.from_prefix.size()));
                return out;
            }
            return std::string(name);
        }

        Status import_weights(const std::filesystem::path& path, NN::ModelState& state,
                              SushiRuntime::span<const RenameRule> rules)
        {
            std::ifstream in(path, std::ios::binary);
            SA_FAIL_IF(!in, StatusCode::FileNotFound, "could not open '{}' for reading",
                       path.string());

            SA_TRY_ASSIGN(CheckpointReader reader, CheckpointReader::open(in));

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

    } // namespace IO
} // namespace SushiAI
