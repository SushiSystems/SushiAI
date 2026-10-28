/**************************************************************************/
/* importer_registry.hpp                                                  */
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
 * @file importer_registry.hpp
 * @brief Declares the importer registry, its rows and the import of a model by format name.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/README.md
 */

#pragma once

#include <filesystem>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/io/imported_model.hpp>

namespace SushiAI
{
    namespace IO
    {
        /**
         * @brief Reads the files of one foreign format into an imported model.
         * @param files The paths in the order the format states them; the importer refuses a
         *              count it does not take.
         * @return The model, or the failure that kept it from being read.
         */
        using ImporterFunction =
            Result<ImportedModel> (*)(SushiRuntime::span<const std::filesystem::path> files);

        /**
         * @struct ImporterEntry
         * @brief One row of an ImporterRegistry.
         */
        struct ImporterEntry
        {
            /** @brief The format name `sushiai import` is given, such as "darknet". */
            std::string_view format;

            /** @brief What reads the files of the format. */
            ImporterFunction read;
        };

        /** @brief Holds the importer rows `sushiai import` may name, by format. */
        class ImporterRegistry final
        {
            public:
                /**
                 * @brief Appends an importer row.
                 * @param row Its format text must outlive the registry; an empty format or a
                 *            null function is fatal.
                 * @return InvalidDescription when a row holds the format.
                 */
                [[nodiscard]] Status add(const ImporterEntry& row) noexcept;

                /** @brief Returns the row of @p format, or null when none holds it. */
                [[nodiscard]] const ImporterEntry* find(std::string_view format) const noexcept;

                /**
                 * @brief Returns the rows in the order they were added.
                 * @return A view the next add() invalidates.
                 */
                [[nodiscard]] SushiRuntime::span<const ImporterEntry> rows() const noexcept;

            private:
                /** @brief Holds the rows in insertion order. */
                std::vector<ImporterEntry> rows_;
        };

        /**
         * @brief Reads @p files through the row @p format names.
         * @return What the row returns; InvalidDescription, naming every format the registry
         *         holds, when @p format is no row.
         */
        [[nodiscard]] Result<ImportedModel> import_model(
            const ImporterRegistry& registry, std::string_view format,
            SushiRuntime::span<const std::filesystem::path> files) noexcept;

    } // namespace IO
} // namespace SushiAI
