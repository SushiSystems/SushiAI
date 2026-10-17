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
 * @file csv.hpp
 * @brief Declares the reader of a dataset kept as a header row and numeric columns.
 * @author Mustafa Garip
 */

#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <SushiAI/failure/result.hpp>
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
    [[nodiscard]] Result<SampleList> read_csv(const CSVOptions& options);
} // namespace SushiAI::Vision::Datasets
