/**************************************************************************/
/* caffe_op_relu.cpp                                                      */
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
 * @file caffe_op_relu.cpp
 * @brief Implements the translator of a Caffe ReLU layer.
 * @author Mustafa Garip
 */

#include "caffe_operator_table.hpp"

#include <cstdint>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/interchange/caffe_net.hpp>

#include "caffe_parameters.hpp"

namespace SushiAI
{
    namespace Vision::Importers
    {
        namespace Detail::Caffe
        {
            Status translate_relu(const Interchange::CaffeLayer& layer, std::int64_t index,
                                  TranslationProgress& progress)
            {
                SA_TRY(expect_shape(layer, 1));
                SA_TRY_ASSIGN(BlobValue value, read_bottom(progress, layer, 0));

                const ParameterReader parameters(layer, "relu_param");
                SA_TRY_ASSIGN(const double slope, parameters.read_real("negative_slope", 0.0));

                const std::int64_t sources[] = {value.row};
                if (slope == 0.0)
                {
                    value.row = append_row(progress, "relu", make_attributes({}, {}), sources);
                }
                else
                {
                    value.row = append_row(progress, "leaky_relu",
                                           make_attributes({"slope"}, {Config::Value::real(slope)}),
                                           sources);
                }
                write_blob(progress, layer.tops.front(), value, index);
                return Status();
            }
        } // namespace Detail::Caffe
    } // namespace Vision::Importers
} // namespace SushiAI
