/**************************************************************************/
/* row_index.cpp                                                          */
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
 * @file row_index.cpp
 * @brief Implements the lowering rows that move rows by an index: gather, scatter-add, take, put and argmax.
 * @author Mustafa Garip
 */

#include <SushiAI/graph/lowering/row_index.hpp>

#include <cstdint>
#include <SushiBLAS/gather/gather.hpp>
#include <SushiBLAS/reductions/reductions.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/lowering/context.hpp>
#include "tensor_views.hpp"

namespace SushiAI
{
    namespace Graph
    {
        namespace
        {
            /**
             * @brief Returns @p tensor seen as a list of @p count elements, over the same storage.
             * @pre @p tensor holds @p count elements; a strided one is fatal.
             */
            [[nodiscard]] SushiBLAS::Tensor as_list(const LoweringContext& context,
                                                    const SushiBLAS::Tensor& tensor,
                                                    std::int64_t count)
            {
                return Detail::view_as(context, tensor,
                                       SushiRuntime::span<const std::int64_t>(&count, 1));
            }

            /**
             * @brief Returns @p tensor seen as @p rows by @p width, over the same storage.
             * @pre @p tensor holds rows * width elements; a strided one is fatal.
             */
            [[nodiscard]] SushiBLAS::Tensor as_rows(const LoweringContext& context,
                                                    const SushiBLAS::Tensor& tensor,
                                                    std::int64_t rows, std::int64_t width)
            {
                const std::int64_t dims[2] = {rows, width};
                return Detail::view_as(context, tensor,
                                       SushiRuntime::span<const std::int64_t>(dims, 2));
            }
        } // namespace

        Status lower_gather_rows(const LoweringContext& context) noexcept
        {
            context.require_arity(2, 1);

            const Value& table_value = context.input_value(0);
            SA_FATAL_IF(table_value.shape.rank() != 2,
                        "GATHER_ROWS (node {}) needs a rank-2 table; got {}", context.id(),
                        table_value.shape.to_string());

            const SushiBLAS::Tensor& table = context.input(0);
            const SushiBLAS::Tensor& indices = context.input(1);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& gathered, context.output(0));

            // The index may have any rank; the gather reads it as a list of one row per entry.
            const std::int64_t count = indices.num_elements;
            const SushiBLAS::Tensor list = as_list(context, indices, count);
            SushiBLAS::Tensor rows = as_rows(context, gathered, count, table_value.shape[1]);

            SushiBLAS::RowIndexOps index_ops(context.recorder());
            index_ops.take_rows(table, list, rows);
            context.count(1);
            return Status();
        }

        Status lower_scatter_add_rows(const LoweringContext& context) noexcept
        {
            context.require_arity(2, 1);

            const Value& sums_value = context.output_value(0);
            SA_FATAL_IF(sums_value.shape.rank() != 2,
                        "SCATTER_ADD_ROWS (node {}) needs a rank-2 result; got {}", context.id(),
                        sums_value.shape.to_string());

            const SushiBLAS::Tensor& source = context.input(0);
            const SushiBLAS::Tensor& indices = context.input(1);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& sums, context.output(0));

            // The rows have any rank above one; the scatter reads them as one row per index.
            const std::int64_t count = indices.num_elements;
            const SushiBLAS::Tensor list = as_list(context, indices, count);
            const SushiBLAS::Tensor rows = as_rows(context, source, count, sums_value.shape[1]);

            // A beta of zero writes every element of the result, so nothing zeroes it first.
            SushiBLAS::RowIndexOps index_ops(context.recorder());
            index_ops.add_rows(rows, list, sums, 0.0);
            context.count(1);
            return Status();
        }

        Status lower_take_along(const LoweringContext& context) noexcept
        {
            context.require_arity(2, 1);

            const SushiBLAS::Tensor& values = context.input(0);
            const SushiBLAS::Tensor& index = context.input(1);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& picked, context.output(0));

            SushiBLAS::RowIndexOps index_ops(context.recorder());
            index_ops.take_along(values, index, picked);
            context.count(1);
            return Status();
        }

        Status lower_put_along(const LoweringContext& context) noexcept
        {
            context.require_arity(2, 1);

            const SushiBLAS::Tensor& picked = context.input(0);
            const SushiBLAS::Tensor& index = context.input(1);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& values, context.output(0));

            SushiBLAS::RowIndexOps index_ops(context.recorder());
            index_ops.put_along(picked, index, values);
            context.count(1);
            return Status();
        }

        Status lower_argmax_rows(const LoweringContext& context) noexcept
        {
            context.require_arity(1, 1);

            const Value& x_value = context.input_value(0);
            SA_FATAL_IF(x_value.shape.rank() != 2,
                        "ARGMAX_ROWS (node {}) needs a rank-2 operand; got {}", context.id(),
                        x_value.shape.to_string());

            const SushiBLAS::Tensor& x = context.input(0);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& columns, context.output(0));

            // Axis 1 reduces each row to the column of its largest element.
            constexpr int K_ROW_AXIS = 1;
            SushiBLAS::ReductionOps reductions(context.recorder());
            reductions.argmax_along(x, K_ROW_AXIS, columns);
            context.count(1);
            return Status();
        }
    } // namespace Graph
} // namespace SushiAI
