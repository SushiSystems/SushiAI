/**************************************************************************/
/* caffe_import.hpp                                                       */
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
 * @file caffe_import.hpp
 * @brief Declares the Caffe importer: a prototxt and a caffemodel in, an imported model out.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/README.md
 */

#pragma once

#include <filesystem>
#include <string_view>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/interchange/caffe_net.hpp>
#include <SushiAI/io/imported_model.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        /** @brief Holds the format name `sushiai import` gives the Caffe importer. */
        inline constexpr std::string_view K_CAFFE_FORMAT = "caffe";

        /**
         * @brief Translates the layers of @p net into model rows, and its blobs into weights.
         * @param origin Names the net in a message.
         * @return Unsupported listing every layer whose type has no row, or the first layer whose
         *         parameters no row can state; InvalidDescription for a malformed net;
         *         Incompatible for blobs that do not fit their layer.
         */
        [[nodiscard]] Result<IO::ImportedModel> import_caffe_net(const Interchange::CaffeNet& net,
                                                                 std::string_view origin) noexcept;

        /**
         * @brief Reads the prototxt of @p files, and the caffemodel when a second file is given,
         *        into an imported model.
         * @return InvalidDescription for a count of files other than one or two; FileNotFound or
         *         InvalidFile for a file that cannot be read; else what read_caffe_prototxt,
         *         read_caffe_model, attach_caffe_weights and import_caffe_net return.
         */
        [[nodiscard]] Result<IO::ImportedModel> import_caffe_files(
            SushiRuntime::span<const std::filesystem::path> files) noexcept;
    } // namespace Vision::Importers
} // namespace SushiAI
