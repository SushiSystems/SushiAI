/**************************************************************************/
/* row_mask.cpp                                                           */
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
 * @file row_mask.cpp
 * @brief Implements the row mask: its key, its leaf, its weights and its weighted loss.
 * @author Mustafa Garip
 */

#include <SushiAI/train/row_mask.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/train/batch_role.hpp>
#include <SushiAI/train/dense_losses.hpp>
#include <SushiAI/train/objective.hpp>

namespace SushiAI
{
    namespace Train
    {
        namespace
        {
            /** @brief Writes each row's weight into the @p width elements of its row of @p to. */
            template<typename Real>
            void write_row_weights(const std::vector<double>& row_weight, std::int64_t width,
                                   Real* to) noexcept
            {
                for (std::size_t row = 0; row < row_weight.size(); ++row)
                {
                    for (std::int64_t column = 0; column < width; ++column)
                        to[static_cast<std::int64_t>(row) * width + column] =
                            static_cast<Real>(row_weight[row]);
                }
            }
        } // namespace

        RowMask::RowMask(std::int64_t columns, std::int64_t column) noexcept
            : active_(true), columns_(columns), column_(column)
        {

        }

        Result<RowMask> RowMask::make(std::int64_t columns, std::int64_t column) noexcept
        {
            SA_FAIL_IF(columns < 0, StatusCode::InvalidDescription,
                       "A row mask has 0 columns (one value per sample) or at least 1; got {}",
                       columns);
            SA_FAIL_IF(column < 0 || column >= std::max<std::int64_t>(columns, 1),
                       StatusCode::InvalidDescription,
                       "A row mask of {} columns has no column {}", columns, column);

            return RowMask(columns, column);
        }

        Result<RowMask> RowMask::read(const Config::Value& attributes,
                                      std::string_view context) noexcept
        {
            if (!attributes.has(K_TYPE_MASK_KEY))
                return RowMask();

            SA_TRY_ASSIGN(const Config::Value& stated, attributes.at(K_TYPE_MASK_KEY, context));
            if (stated.kind() == Config::Kind::Bool)
            {
                SA_TRY_ASSIGN(const bool on, stated.as_bool(context));
                if (!on)
                    return RowMask();

                return make(0, 0);
            }

            SA_FAIL_IF(stated.kind() != Config::Kind::Object, StatusCode::InvalidDescription,
                       "{}: '{}' is a boolean or an object of '{}' and '{}', but the file "
                       "holds a {}",
                       context, K_TYPE_MASK_KEY, K_TYPE_MASK_COLUMNS_KEY, K_TYPE_MASK_COLUMN_KEY,
                       Config::kind_name(stated.kind()));

            const std::array<std::string_view, 2> allowed{K_TYPE_MASK_COLUMNS_KEY,
                                                          K_TYPE_MASK_COLUMN_KEY};
            SA_TRY(stated.reject_unknown_keys(
                context, SushiRuntime::span<const std::string_view>(allowed.data(),
                                                                    allowed.size())));
            SA_FAIL_IF(!stated.has(K_TYPE_MASK_COLUMNS_KEY), StatusCode::InvalidDescription,
                       "{}: '{}' as an object states '{}', the columns of the mask role",
                       context, K_TYPE_MASK_KEY, K_TYPE_MASK_COLUMNS_KEY);

            SA_TRY_ASSIGN(const Config::Value& columns_value,
                          stated.at(K_TYPE_MASK_COLUMNS_KEY, context));
            SA_TRY_ASSIGN(const std::int64_t columns, columns_value.as_int(context));
            SA_FAIL_IF(columns < 1, StatusCode::InvalidDescription,
                       "{}: '{}' of '{}' is at least 1; got {}", context,
                       K_TYPE_MASK_COLUMNS_KEY, K_TYPE_MASK_KEY, columns);

            std::int64_t column = 0;
            if (stated.has(K_TYPE_MASK_COLUMN_KEY))
            {
                SA_TRY_ASSIGN(const Config::Value& column_value,
                              stated.at(K_TYPE_MASK_COLUMN_KEY, context));
                SA_TRY_ASSIGN(column, column_value.as_int(context));
            }

            return make(columns, column);
        }

        void RowMask::declare(const Graph::GraphBuilder& builder, ObjectiveLeaves& leaves,
                              const Shape& element_shape, DType dtype) const
        {
            if (!active_)
                return;

            leaves.add(K_TYPE_MASK_WEIGHTS_LEAF,
                       builder.input(element_shape, K_TYPE_MASK_WEIGHTS_LEAF, dtype));
            leaves.require(Data::TargetDescription{std::string(Data::K_ROLE_TYPE_MASK),
                                                   columns_ == 0 ? Shape{} : Shape{columns_},
                                                   DType::Float32, 0});
        }

        Result<KeptRows> RowMask::kept_rows(const Data::Batch& batch) const noexcept
        {
            if (!active_)
                return KeptRows();

            SA_TRY_ASSIGN(const Data::BatchTarget* const mask,
                          require_batch_role(batch, Data::K_ROLE_TYPE_MASK));

            const SushiBLAS::Tensor& values = mask->values;
            const std::int64_t stride = std::max<std::int64_t>(columns_, 1);
            SA_FAIL_IF(values.dtype != DType::Float32, StatusCode::Incompatible,
                       "The role '{}' is {} but the mask reads {}", Data::K_ROLE_TYPE_MASK,
                       dtype_name(values.dtype), dtype_name(DType::Float32));
            SA_FAIL_IF(values.num_elements < 1 || values.num_elements % stride != 0,
                       StatusCode::Incompatible,
                       "The role '{}' holds {} values, which are no rows of {} columns",
                       Data::K_ROLE_TYPE_MASK, values.num_elements, stride);

            const std::int64_t rows = values.num_elements / stride;
            const float* const stated = values.data_as<float>();
            for (std::int64_t row = 0; row < rows; ++row)
            {
                const float value = stated[row * stride + column_];
                SA_FAIL_IF(value != 0.0f && value != 1.0f, StatusCode::Incompatible,
                           "The mask of row {} is {}; a mask is 0 or 1", row, value);
            }

            std::vector<std::uint8_t> flags;
            SA_TRY(SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate,
                                          [&]() -> Status
            {
                flags.reserve(static_cast<std::size_t>(rows));
                for (std::int64_t row = 0; row < rows; ++row)
                    flags.push_back(stated[row * stride + column_] == 1.0f ? 1 : 0);
                return Status();
            }));

            return KeptRows(std::move(flags));
        }

        Result<KeptRows> RowMask::kept_rows_of(const Data::Batch& batch,
                                               std::int64_t rows) const noexcept
        {
            SA_TRY_ASSIGN(KeptRows kept, kept_rows(batch));
            SA_FAIL_IF(active_ && kept.size() != rows, StatusCode::Incompatible,
                       "The role '{}' holds {} rows but the objective reads {}",
                       Data::K_ROLE_TYPE_MASK, kept.size(), rows);

            return kept;
        }

        Status RowMask::feed(const Data::Batch& batch, const ObjectiveLeaves& leaves,
                             Graph::TensorPool& pool) const
        {
            if (!active_)
                return Status();

            SA_TRY_ASSIGN(const KeptRows kept, kept_rows(batch));
            SA_TRY_ASSIGN(SushiBLAS::Tensor* const weights,
                          find_leaf_tensor(leaves, K_TYPE_MASK_WEIGHTS_LEAF, pool));

            const std::int64_t rows = kept.size();
            SA_FAIL_IF(weights->num_elements % rows != 0, StatusCode::Incompatible,
                       "The role '{}' holds {} rows but the weights leaf holds {} elements",
                       Data::K_ROLE_TYPE_MASK, rows, weights->num_elements);
            SA_FAIL_IF(weights->dtype != DType::Float32 && weights->dtype != DType::Float64,
                       StatusCode::Incompatible,
                       "The weights leaf is {}; weights are written as {} or {}",
                       dtype_name(weights->dtype), dtype_name(DType::Float32),
                       dtype_name(DType::Float64));

            // A batch that keeps no row gets weights of 0, so its loss and gradient are 0.
            const double share = 1.0 / static_cast<double>(std::max<std::int64_t>(kept.count(), 1));
            std::vector<double> row_weight;
            SA_TRY(SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate,
                                          [&]() -> Status
            {
                row_weight.reserve(static_cast<std::size_t>(rows));
                for (std::int64_t row = 0; row < rows; ++row)
                    row_weight.push_back(kept.keeps(row) ? share : 0.0);
                return Status();
            }));

            const std::int64_t width = weights->num_elements / rows;
            if (weights->dtype == DType::Float64)
                write_row_weights(row_weight, width, weights->data_as<double>());
            else
                write_row_weights(row_weight, width, weights->data_as<float>());

            return Status();
        }

        Graph::Traced RowMask::emit_weighted(const Graph::GraphBuilder& builder,
                                             const Graph::Graph& graph,
                                             const Graph::Traced& terms,
                                             const ObjectiveLeaves& leaves,
                                             double row_scale) const noexcept
        {
            SA_TRY_ASSIGN(const Graph::ValueID held, terms);

            const Graph::Traced weighted =
                builder.mul(held, leaves.at(K_TYPE_MASK_WEIGHTS_LEAF));

            return builder.scale(emit_sum_of_elements(builder, graph, weighted), row_scale);
        }

    } // namespace Train
} // namespace SushiAI
