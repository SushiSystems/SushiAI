/**************************************************************************/
/* caffe_writer.hpp                                                       */
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
 * @file caffe_writer.hpp
 * @brief Declares the Caffe prototxt and caffemodel writers, inverses of the two readers.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/README.md
 */

#pragma once

#include <cstddef>
#include <string>
#include <vector>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/interchange/caffe_net.hpp>

namespace SushiAI
{
    namespace Interchange
    {
        /**
         * @brief Writes the structure of @p net, without blobs, as a prototxt.
         * Each layer is a V2 `layer` with name, type, bottoms, tops and its other parameter
         * fields; each input is an `input` and an `input_shape`.
         * @return The text; InvalidDescription naming a layer with no name or type, or an input
         *         with no shape.
         */
        [[nodiscard]] Result<std::string> write_caffe_prototxt(const CaffeNet& net) noexcept;

        /**
         * @brief Writes the layers of @p net that hold blobs as a caffemodel.
         *
         * Each such layer is written as a V2 `layer` with its name, type, bottoms, tops and
         * blobs, each blob with a `shape` and packed float32 `data`.
         * @return The file; InvalidDescription naming a blob whose bytes do not fill its
         *         extents.
         */
        [[nodiscard]] Result<std::vector<std::byte>> write_caffe_model(
            const CaffeNet& net) noexcept;
    } // namespace Interchange
} // namespace SushiAI
