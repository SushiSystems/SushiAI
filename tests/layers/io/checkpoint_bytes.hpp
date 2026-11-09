/**************************************************************************/
/* checkpoint_bytes.hpp                                                   */
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
 * @file checkpoint_bytes.hpp
 * @brief Declares the helpers that write checkpoint bytes in memory for the io tests.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/checkpoint_format.hpp>
#include <SushiAI/nn/model_state.hpp>

namespace SushiAI
{
    namespace Test
    {
        /**
         * @brief Writes a checkpoint of @p records to @p out, every payload byte zero.
         * @param out The stream the checkpoint goes to.
         * @param records The records, in the order they are written.
         */
        inline void write_zero_checkpoint(std::ostream& out,
                                          const std::vector<IO::TensorRecord>& records)
        {
            SA_ASSERT_OK_AND_ASSIGN(
                IO::CheckpointWriter writer,
                IO::CheckpointWriter::create(out, static_cast<std::uint32_t>(records.size())));
            for (const IO::TensorRecord& record : records)
            {
                const std::vector<std::uint8_t> payload(record.payload_bytes(), 0);
                SA_EXPECT_OK(writer.write(record, payload.data()));
            }
            writer.finish();
        }

        /** @brief Returns the bytes of a checkpoint of @p records with zero payloads. */
        inline std::string checkpoint_bytes(const std::vector<IO::TensorRecord>& records)
        {
            std::ostringstream out;
            write_zero_checkpoint(out, records);
            return out.str();
        }

        /** @brief Views @p bytes as the span a loader reads. */
        inline SushiRuntime::span<const std::uint8_t> as_bytes(const std::string& bytes)
        {
            return SushiRuntime::span<const std::uint8_t>(
                reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size());
        }

        /** @brief Returns one record per entry of @p state, in its order, or per parameter only. */
        inline std::vector<IO::TensorRecord> records_of(NN::ModelState& state,
                                                        bool parameters_only = false)
        {
            std::vector<IO::TensorRecord> records;
            for (const NN::StateEntry& entry : state.entries())
            {
                if (parameters_only && entry.kind != NN::StateKind::Parameter)
                    continue;
                IO::TensorRecord record;
                record.name = entry.name;
                record.dtype = entry.dtype;
                record.shape = entry.shape;
                records.push_back(record);
            }
            return records;
        }
    } // namespace Test
} // namespace SushiAI
