/**************************************************************************/
/* exporter_registry.hpp                                                  */
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
 * @file exporter_registry.hpp
 * @brief Declares the exporter registry, its rows and the export of a model by format name.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/README.md
 */

#pragma once

#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/io/exported_model.hpp>
#include <SushiAI/io/imported_model.hpp>

namespace SushiAI
{
    namespace IO
    {
        /** @brief Holds the word `sushiai export` takes for the weight formats; no row takes it. */
        inline constexpr std::string_view K_WEIGHT_EXPORT_WORD = "weights";

        /**
         * @brief Writes a model, its weights and its training recipe as the files of one format.
         * @param model The model as `read_model_files` reads it.
         * @param stem The name every written file starts with.
         * @return The files and the notes; Unsupported naming each row the format cannot state.
         */
        using ExporterFunction = Result<ExportedModel> (*)(const ImportedModel& model,
                                                           std::string_view stem);

        /**
         * @struct ExporterEntry
         * @brief One row of an ExporterRegistry.
         */
        struct ExporterEntry
        {
            /** @brief The format name `sushiai export` is given, such as "onnx". */
            std::string_view format;

            /** @brief What writes the files of the format. */
            ExporterFunction write;
        };

        /** @brief Holds the exporter rows `sushiai export` may name, by format. */
        class ExporterRegistry final
        {
            public:
                /**
                 * @brief Appends an exporter row.
                 * @param row Its format text must outlive the registry; an empty format or a
                 *            null function is fatal.
                 * @return InvalidDescription when a row holds the format or the format is
                 *         K_WEIGHT_EXPORT_WORD.
                 */
                [[nodiscard]] Status add(const ExporterEntry& row) noexcept;

                /** @brief Returns the row of @p format, or null when none holds it. */
                [[nodiscard]] const ExporterEntry* find(std::string_view format) const noexcept;

                /**
                 * @brief Returns the rows in the order they were added.
                 * @return A view the next add() invalidates.
                 */
                [[nodiscard]] SushiRuntime::span<const ExporterEntry> rows() const noexcept;

            private:
                /** @brief Holds the rows in insertion order. */
                std::vector<ExporterEntry> rows_;
        };

        /**
         * @brief Writes @p model through the row @p format names.
         * @return What the row returns; InvalidDescription, naming every format the registry
         *         holds, when @p format is no row.
         */
        [[nodiscard]] Result<ExportedModel> export_model(const ExporterRegistry& registry,
                                                         std::string_view format,
                                                         const ImportedModel& model,
                                                         std::string_view stem) noexcept;

    } // namespace IO
} // namespace SushiAI
