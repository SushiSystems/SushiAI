/**************************************************************************/
/* darknet_export.hpp                                                     */
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
 * @file darknet_export.hpp
 * @brief Declares the Darknet exporter, which writes a model as a `.cfg` and a `.weights`.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/EXPORT.md
 */

#pragma once

#include <string_view>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/io/exported_model.hpp>
#include <SushiAI/io/imported_model.hpp>

namespace SushiAI
{
    namespace Vision::Exporters
    {
        /** @brief Holds the format name `sushiai export` gives the Darknet exporter. */
        inline constexpr std::string_view K_DARKNET_EXPORT_FORMAT = "darknet";

        /**
         * @brief Writes @p model as `<stem>.cfg`, and `<stem>.weights` when it has weights, the
         *        exporter row of `darknet`.
         * @return The files and a note for each value they leave out; Unsupported naming each
         *         refused row as `row <i> '<type>': <reason>`, or the model's inputs, input
         *         type, input rank, fit, task or outputs.
         */
        [[nodiscard]] Result<IO::ExportedModel> export_darknet_files(
            const IO::ImportedModel& model, std::string_view stem) noexcept;
    } // namespace Vision::Exporters
} // namespace SushiAI
