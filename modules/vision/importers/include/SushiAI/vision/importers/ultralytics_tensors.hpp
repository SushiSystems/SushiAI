/**************************************************************************/
/* ultralytics_tensors.hpp                                                */
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
 * @file ultralytics_tensors.hpp
 * @brief Declares the weight records of an Ultralytics checkpoint in SushiAI's layouts.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/ULTRALYTICS.md
 */

#pragma once

#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/vision/importers/ultralytics_checkpoint.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        /**
         * @brief Returns each tensor of @p tensors as a float32 weight record of the same name.
         *
         * A convolution weight [O, I, kh, kw] becomes [O, kh, kw, I]; a depthwise one
         * [C, 1, kh, kw] becomes [kh, kw, C]. Every other tensor keeps its layout.
         * @return Unsupported naming a grouped convolution that is not depthwise.
         */
        [[nodiscard]] Result<std::vector<IO::ImportedTensor>> list_ultralytics_tensors(
            SushiRuntime::span<const UltralyticsTensor> tensors) noexcept;
    } // namespace Vision::Importers
} // namespace SushiAI
