/**************************************************************************/
/* byte_sink.cpp                                                          */
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
 * @file byte_sink.cpp
 * @brief Implements the sink that writes to memory and the one that writes to the file system.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/datasets/byte_sink.hpp>

#include <fstream>
#include <string>
#include <system_error>

#include "disk.hpp"

namespace SushiAI::Vision::Datasets::Detail
{
    namespace
    {
        /** @brief Holds what a file is called while it is being written. */
        constexpr std::string_view K_PARTIAL_EXTENSION = ".partial";
    } // namespace

    Status FileSystemSink::write(const std::filesystem::path& file, std::string_view bytes,
                                 std::string_view reader) const
    {
        return at_boundary(reader, [&]() -> Status
        {
            std::filesystem::create_directories(file.parent_path());

            std::filesystem::path partial = file;
            partial += std::string(K_PARTIAL_EXTENSION);

            {
                std::ofstream out(partial, std::ios::binary | std::ios::trunc);
                SA_FAIL_IF(!out.is_open(), StatusCode::InvalidFile,
                           "'{}': {}: the file cannot be opened for writing", display_of(partial),
                           reader);

                out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
                out.flush();
                SA_FAIL_IF(!out, StatusCode::InvalidFile, "'{}': {}: writing the file failed",
                           display_of(partial), reader);
            }

            // A file under the final name is whole: the rename is the last step.
            std::filesystem::rename(partial, file);
            return Status();
        }, file);
    }

    const ByteSink& file_sink() noexcept
    {
        static const FileSystemSink K_FILE_SINK;
        return K_FILE_SINK;
    }
} // namespace SushiAI::Vision::Datasets::Detail

namespace SushiAI::Vision::Datasets
{
    MemorySink::MemorySink(MemorySource& target) noexcept : target_(&target)
    {
    }

    Status MemorySink::write(const std::filesystem::path& file, std::string_view bytes,
                             std::string_view reader) const
    {
        return Detail::at_boundary(reader, [&]() -> Status
        {
            target_->add_file(file, std::string(bytes));
            return Status();
        }, file);
    }
} // namespace SushiAI::Vision::Datasets
