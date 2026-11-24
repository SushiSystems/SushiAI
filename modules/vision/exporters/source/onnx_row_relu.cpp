/**************************************************************************/
/* onnx_row_relu.cpp                                                      */
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
 * @file onnx_row_relu.cpp
 * @brief Defines the writer of the "relu" row, one ONNX Relu node.
 * @author Mustafa Garip
 */

#include <SushiAI/failure/status.hpp>
#include "onnx_export_progress.hpp"
#include "onnx_row_table.hpp"

namespace SushiAI
{
    namespace Vision::Exporters
    {
        namespace Detail
        {
            Status write_onnx_relu(const OnnxRowCall& call, OnnxExportProgress& progress)
            {
                return write_pointwise_row(call, progress, "Relu");
            }
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
