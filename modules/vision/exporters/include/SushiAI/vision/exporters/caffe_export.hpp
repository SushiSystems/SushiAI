/**************************************************************************/
/* caffe_export.hpp                                                       */
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
 * @file caffe_export.hpp
 * @brief Declares the Caffe exporter, which writes a model as a `.prototxt` and a `.caffemodel`.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/EXPORT.md
 */

#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/interchange/caffe_net.hpp>
#include <SushiAI/io/exported_model.hpp>
#include <SushiAI/io/imported_model.hpp>

namespace SushiAI
{
    namespace Vision::Exporters
    {
        /** @brief Holds the format name `sushiai export` gives the Caffe exporter. */
        inline constexpr std::string_view K_CAFFE_EXPORT_FORMAT = "caffe";

        /**
         * @brief Returns the net of @p model: an Input layer `input` [1, C, H, W] or [1, K],
         *        then layers `row_<i>`, each blob holding its weights when the model has some.
         * @return Unsupported naming each refused row as `row <i> '<type>': <reason>` and each
         *         output the importer would not publish back, or the model's inputs or input.
         */
        [[nodiscard]] Result<Interchange::CaffeNet> export_caffe_net(
            const IO::ImportedModel& model) noexcept;

        /** @brief Returns one note for each value of @p model the Caffe files do not hold. */
        [[nodiscard]] std::vector<std::string> describe_caffe_losses(
            const IO::ImportedModel& model) noexcept;

        /**
         * @brief Writes @p model as `<stem>.prototxt` and, when it has weights,
         *        `<stem>.caffemodel`; the exporter row of `caffe`.
         * @return What export_caffe_net and the Caffe writers return, and the notes.
         */
        [[nodiscard]] Result<IO::ExportedModel> export_caffe_files(const IO::ImportedModel& model,
                                                                   std::string_view stem) noexcept;
    } // namespace Vision::Exporters
} // namespace SushiAI
