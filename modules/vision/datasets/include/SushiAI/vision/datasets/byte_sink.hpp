/**************************************************************************/
/* byte_sink.hpp                                                          */
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
 * @file byte_sink.hpp
 * @brief Declares the port the sample cache writes its files through.
 * @author Mustafa Garip
 */

#pragma once

#include <string_view>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/datasets/byte_source.hpp>

/** @brief Holds the readers of the on-disk layouts vision datasets are kept in. */
namespace SushiAI::Vision::Datasets
{
    /**
     * @class ByteSink
     * @brief Takes the bytes of the files a writer produces, beside the ByteSource it reads.
     *
     * A writer that takes a ByteSink writes to the file system, through the overload that takes
     * no sink, and to memory, through MemorySink. A file under a name is whole: a sink replaces
     * a file at once or leaves it as it was.
     */
    class ByteSink
    {
        public:
            virtual ~ByteSink() = default;

            /**
             * @brief Writes a file, creating the folders above it and replacing a file of that name.
             * @param reader Names the calling writer in a message.
             * @return InvalidFile, opening with the path, when the file cannot be written.
             */
            [[nodiscard]] virtual Status write(const std::filesystem::path& file,
                                               std::string_view bytes,
                                               std::string_view reader) const = 0;
    };

    /**
     * @class MemorySink
     * @brief Writes the files into a MemorySource, so a ByteSource over it reads them back.
     */
    class MemorySink final : public ByteSink
    {
        public:
            /** @pre @p target outlives the sink. */
            explicit MemorySink(MemorySource& target) noexcept;

            [[nodiscard]] Status write(const std::filesystem::path& file, std::string_view bytes,
                                       std::string_view reader) const override;

        private:
            /** @brief Receives the files. */
            MemorySource* target_;
    };

    /**
     * @brief Holds where a dataset build reads its files from and writes its cache files to.
     *
     * Both null is the file system, which is what a default-constructed value holds. A test
     * serves the whole build from a MemorySource and a MemorySink over it.
     * @pre What the value points at outlives every call and every dataset it is passed to.
     */
    struct DatasetStorage
    {
        /** @brief Serves the folders, the files and the cache files that exist; null is the file system. */
        const ByteSource* source = nullptr;

        /** @brief Takes the cache files; null is the file system. */
        const ByteSink* sink = nullptr;
    };
} // namespace SushiAI::Vision::Datasets
