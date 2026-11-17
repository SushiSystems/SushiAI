/**************************************************************************/
/* caffe_op_eltwise.cpp                                                   */
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
 * @file caffe_op_eltwise.cpp
 * @brief Implements the translator of a Caffe Eltwise layer: a chain of add, mul or maximum rows.
 * @author Mustafa Garip
 */

#include "caffe_operator_table.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
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
            namespace
            {
                /** @brief Holds the fewest bottoms Caffe's Eltwise layer takes. */
                constexpr std::size_t K_FEWEST_BOTTOMS = 2;

                /**
                 * @brief Reads the operation of @p layer and checks its coefficients.
                 * @return The nn row of SUM, PROD or MAX: add, mul or maximum; Unsupported for a coefficient other than 1
                 *         on a sum; InvalidDescription for any coefficient on a product or a
                 *         maximum, which Caffe itself refuses, and for an operation Caffe does not
                 *         know.
                 */
                [[nodiscard]] Result<std::string_view> read_operation(
                    const Interchange::CaffeLayer& layer)
                {
                    const ParameterReader parameters(layer, "eltwise_param");
                    SA_TRY_ASSIGN(const std::string operation,
                                  parameters.read_word("operation", "SUM"));
                    SA_FAIL_IF(operation != "SUM" && operation != "PROD" && operation != "MAX",
                               StatusCode::InvalidDescription,
                               "{} states the operation {}; Caffe knows SUM, PROD and MAX",
                               describe_layer(layer), operation);

                    SA_TRY_ASSIGN(const std::vector<double> coefficients,
                                  parameters.read_reals("coeff"));
                    SA_FAIL_IF(operation != "SUM" && !coefficients.empty(),
                               StatusCode::InvalidDescription,
                               "{} states coefficients for the operation {}; Caffe takes them "
                               "for SUM only",
                               describe_layer(layer), operation);
                    for (const double coefficient : coefficients)
                    {
                        SA_FAIL_IF(coefficient != 1.0, StatusCode::Unsupported,
                                   "{} weights a bottom by {}; the add row sums with weight 1",
                                   describe_layer(layer), coefficient);
                    }

                    if (operation == "PROD")
                        return std::string_view("mul");
                    if (operation == "MAX")
                        return std::string_view("maximum");

                    return std::string_view("add");
                }
            } // namespace

            Status translate_eltwise(const Interchange::CaffeLayer& layer, std::int64_t index,
                                     TranslationProgress& progress)
            {
                SA_FAIL_IF(layer.bottoms.size() < K_FEWEST_BOTTOMS || layer.tops.size() != 1,
                           StatusCode::InvalidDescription,
                           "{} has {} bottoms and {} tops; this type takes two or more and 1",
                           describe_layer(layer), layer.bottoms.size(), layer.tops.size());
                SA_TRY_ASSIGN(const std::string_view row_type, read_operation(layer));

                SA_TRY_ASSIGN(BlobValue sum, read_bottom(progress, layer, 0));
                SA_TRY(expect_image(layer, sum));
                for (std::size_t position = 1; position < layer.bottoms.size(); ++position)
                {
                    SA_TRY_ASSIGN(const BlobValue term, read_bottom(progress, layer, position));
                    SA_TRY(expect_image(layer, term));
                    SA_FAIL_IF(term.height != sum.height || term.width != sum.width ||
                                   term.channels != sum.channels,
                               StatusCode::InvalidDescription,
                               "{} combines a {} by {} by {} blob and a {} by {} by {} one",
                               describe_layer(layer), sum.height, sum.width, sum.channels,
                               term.height, term.width, term.channels);

                    const std::int64_t sources[] = {sum.row, term.row};
                    sum.row = append_row(progress, row_type, make_attributes({}, {}), sources);
                }
                write_blob(progress, layer.tops.front(), sum, index);
                return Status();
            }
        } // namespace Detail::Caffe
    } // namespace Vision::Importers
} // namespace SushiAI
