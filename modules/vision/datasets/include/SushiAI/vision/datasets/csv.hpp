/**************************************************************************/
/* csv.hpp                                                                */
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
 * @file csv.hpp
 * @brief Declares the reader of a dataset kept as a header row and numeric columns.
 * @author Mustafa Garip
 */

#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/vision/datasets/byte_source.hpp>
#include <SushiAI/vision/datasets/sample.hpp>

/** @brief Holds the readers of the on-disk layouts vision datasets are kept in. */
namespace SushiAI::Vision::Datasets
{
    /** @brief Holds what read_csv reads. */
    struct CSVOptions
    {
        /** @brief Holds the directory of the dataset. */
        std::filesystem::path root;

        /** @brief Names the file root/<split>.csv; not read when file is set. */
        std::string split = "train";

        /** @brief Names the file, a relative one taken from root; empty for the split's. */
        std::filesystem::path file;

        /** @brief Names the feature columns; empty for every column that is no target. */
        std::vector<std::string> features;

        /** @brief Names the target columns; empty for the last column. */
        std::vector<std::string> targets;
    };

    /**
     * @brief Reads the feature and the target columns of every record into a ValueTable.
     *
     * A sample is one record; its row of SampleList::table holds the features, then the
     * targets. README.md, section csv.hpp, gives the syntax read and every status.
     * @return The list; a failure that names the file, the line and the column at fault.
     */
    [[nodiscard]] Result<SampleList> read_csv(const CSVOptions& options) noexcept;

    /**
     * @brief Reads the columns of a CSV file served by @p source.
     *
     * read_csv(options) calls this with the file system; a MemorySource serves files
     * that were never written.
     * @return What the overload above returns.
     */
    [[nodiscard]] Result<SampleList> read_csv(const CSVOptions& options,
                                              const ByteSource& source) noexcept;
} // namespace SushiAI::Vision::Datasets
