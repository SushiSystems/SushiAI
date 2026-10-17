/**************************************************************************/
/* linalg.cpp                                                             */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file linalg.cpp
 * @brief Implements the lowering rows of the linear-algebra operations.
 * @author Mustafa Garip
 */

#include <SushiAI/graph/lowering/linalg.hpp>

#include <cstdint>
#include <numeric>
#include <SushiBLAS/blas/blas.hpp>
#include <SushiBLAS/core/sparse_types.hpp>
#include <SushiBLAS/elementwise/elementwise.hpp>
#include <SushiBLAS/sparse/sparse.hpp>
#include <SushiBLAS/sparse_storage/sparse_matrix.hpp>
#include <SushiBLAS/spatial/spatial.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/lowering/context.hpp>

namespace SushiAI
{
    namespace Graph
    {
        namespace
        {
            /** @brief Names the scratch slot that holds a row selection's row offsets. */
            constexpr int K_ROW_OFFSET_SLOT = 0;

            /**
             * @brief Returns @p tensor seen as @p rows by @p width, over the same storage.
             * @pre @p tensor holds rows * width elements; a strided one is fatal.
             */
            [[nodiscard]] SushiBLAS::Tensor as_rows(const LoweringContext& context,
                                                    const SushiBLAS::Tensor& tensor,
                                                    std::int64_t rows, std::int64_t width)
            {
                SA_FATAL_IF(!tensor.is_contiguous(),
                            "{} (node {}) needs contiguous operands; a strided view cannot be "
                            "read as rows",
                            context.name(), context.id());

                const std::int64_t dims[2] = {rows, width};
                return SushiBLAS::Tensor(tensor.storage,
                                         SushiRuntime::span<const std::int64_t>(dims, 2),
                                         tensor.dtype, tensor.storage_offset, tensor.layout);
            }

            /**
             * @brief Builds the CSR matrix whose row n holds a single one, at column indices[n].
             * @param indices The INT32 index operand, of any rank; a strided one is fatal.
             * @param table_rows The column count of the matrix, which is the table's row count.
             * @param dtype The element type of the ones, which is that of the product.
             * @return The [count, table_rows] matrix, or the failure of an allocation.
             */
            [[nodiscard]] Result<SushiBLAS::SparseMatrix> row_selection(
                const LoweringContext& context, const SushiBLAS::Tensor& indices,
                std::int64_t table_rows, DType dtype)
            {
                SA_FATAL_IF(!indices.is_contiguous(),
                            "{} (node {}) needs a contiguous index; a strided view cannot be "
                            "read as a list",
                            context.name(), context.id());

                const std::int64_t count = indices.num_elements;
                SA_TRY_ASSIGN(SushiBLAS::Tensor& offsets,
                              context.scratch(K_ROW_OFFSET_SLOT, Shape{count + 1}, DType::Int32));
                SA_TRY_ASSIGN(SushiBLAS::Tensor& ones, context.ones(Shape{count}, dtype));

                // No task writes the offsets, so the host fills them: row n holds entry n alone.
                std::int32_t* const first = offsets.data_as<std::int32_t>();
                std::iota(first, first + count + 1, std::int32_t{0});

                SushiBLAS::SparseMatrix selection;
                selection.rows = count;
                selection.cols = table_rows;
                selection.index_base = SushiBLAS::Core::IndexBase::Zero;
                selection.row_offsets = offsets;
                selection.column_indices = SushiBLAS::Tensor(
                    indices.storage, SushiRuntime::span<const std::int64_t>(&count, 1),
                    indices.dtype, indices.storage_offset, indices.layout);
                selection.values = ones;
                return selection;
            }

            /** @brief Names the scratch slot of the host-written FLOAT32 keep factors. */
            constexpr int K_HOST_KEEP_SLOT = 0;

            /** @brief Names the scratch slot of the host-written FLOAT32 replacements. */
            constexpr int K_HOST_SHIFT_SLOT = 1;

            /** @brief Names the scratch slot of the keep factors in the operand's type. */
            constexpr int K_KEEP_SLOT = 2;

            /** @brief Names the scratch slot of the replacements in the operand's type. */
            constexpr int K_SHIFT_SLOT = 3;

            /** @brief Holds the scale and the shift that replace the upper triangle. */
            struct TriangleAffine
            {
                SushiBLAS::Tensor* keep = nullptr;  /**< @brief One on and below the diagonal. */
                SushiBLAS::Tensor* shift = nullptr; /**< @brief The fill above the diagonal. */
            };

            /**
             * @brief Builds the [T * T] scale and shift of a causal mask in @p dtype.
             * @param length T, the extent of each of the last two axes.
             * @param fill The value an entry above the diagonal takes.
             * @return The two vectors, or the failure of an allocation.
             */
            [[nodiscard]] Result<TriangleAffine> triangle_affine(const LoweringContext& context,
                                                                 std::int64_t length, double fill,
                                                                 DType dtype)
            {
                const Shape entries{length * length};
                SA_TRY_ASSIGN(SushiBLAS::Tensor& host_keep,
                              context.scratch(K_HOST_KEEP_SLOT, entries, DType::Float32));
                SA_TRY_ASSIGN(SushiBLAS::Tensor& host_shift,
                              context.scratch(K_HOST_SHIFT_SLOT, entries, DType::Float32));

                // No task writes the two vectors, so the host fills them once, at lowering.
                float* const keep = host_keep.data_as<float>();
                float* const shift = host_shift.data_as<float>();
                for (std::int64_t row = 0; row < length; ++row)
                {
                    for (std::int64_t column = 0; column < length; ++column)
                    {
                        const bool kept = column <= row;
                        keep[row * length + column] = kept ? 1.0f : 0.0f;
                        shift[row * length + column] = kept ? 0.0f : static_cast<float>(fill);
                    }
                }

                if (dtype == DType::Float32)
                    return TriangleAffine{&host_keep, &host_shift};

                SA_TRY_ASSIGN(SushiBLAS::Tensor& typed_keep,
                              context.scratch(K_KEEP_SLOT, entries, dtype));
                SA_TRY_ASSIGN(SushiBLAS::Tensor& typed_shift,
                              context.scratch(K_SHIFT_SLOT, entries, dtype));

                SushiBLAS::ElementwiseOps elementwise(context.recorder());
                elementwise.convert(host_keep, typed_keep);
                elementwise.convert(host_shift, typed_shift);
                context.count(2);
                return TriangleAffine{&typed_keep, &typed_shift};
            }

            /**
             * @brief Returns @p tensor seen as [matrices, 1, 1, entries], over the same storage.
             * @pre @p tensor holds matrices * entries elements; a strided one is fatal.
             */
            [[nodiscard]] SushiBLAS::Tensor as_entry_channels(const LoweringContext& context,
                                                              const SushiBLAS::Tensor& tensor,
                                                              std::int64_t matrices,
                                                              std::int64_t entries)
            {
                SA_FATAL_IF(!tensor.is_contiguous(),
                            "{} (node {}) needs contiguous operands; a strided view cannot be "
                            "read matrix by matrix",
                            context.name(), context.id());

                const std::int64_t dims[4] = {matrices, 1, 1, entries};
                return SushiBLAS::Tensor(tensor.storage,
                                         SushiRuntime::span<const std::int64_t>(dims, 4),
                                         tensor.dtype, tensor.storage_offset, tensor.layout);
            }

            /** @brief Stops the process unless the index operand of the node is INT32. */
            void require_int32_index(const LoweringContext& context)
            {
                SA_FATAL_IF(context.input_value(1).dtype != DType::Int32,
                            "{} (node {}) needs an INT32 index; got {}", context.name(),
                            context.id(), dtype_name(context.input_value(1).dtype));
            }
        } // namespace

        Status lower_matmul(const LoweringContext& context)
        {
            context.require_arity(2, 1);

            const SushiBLAS::Tensor& a = context.input(0);
            const SushiBLAS::Tensor& b = context.input(1);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& c, context.output(0));

            SushiBLAS::BLASOps blas(context.recorder());
            blas.gemm(a, b, c, transpose_of(context.node().attributes.flag(0)),
                      transpose_of(context.node().attributes.flag(1)), 1.0, 0.0);
            context.count(1);
            return Status();
        }

        Status lower_bias_add(const LoweringContext& context)
        {
            context.require_arity(2, 1);

            const Value& x_value = context.input_value(0);
            SA_FATAL_IF(x_value.shape.rank() != 2,
                        "BIAS_ADD (node {}) needs a rank-2 left operand; got {}",
                        context.id(), x_value.shape.to_string());

            const SushiBLAS::Tensor& x = context.input(0);
            const SushiBLAS::Tensor& bias = context.input(1);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& y, context.output(0));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& rows,
                          context.ones(Shape{x_value.shape[0]}, x_value.dtype));

            SushiBLAS::BLASOps blas(context.recorder());

            // ger adds the outer product to y, so y holds x before it runs.
            blas.copy(x, y);
            blas.ger(rows, bias, y, 1.0f);
            context.count(2);
            return Status();
        }

        Status lower_sum_rows(const LoweringContext& context)
        {
            context.require_arity(1, 1);

            const Value& x_value = context.input_value(0);
            SA_FATAL_IF(x_value.shape.rank() != 2,
                        "SUM_ROWS needs a rank-2 operand; got {}", x_value.shape.to_string());

            const SushiBLAS::Tensor& x = context.input(0);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& sums, context.output(0));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& ones_rows,
                          context.ones(Shape{x_value.shape[0]}, x_value.dtype));

            SushiBLAS::BLASOps blas(context.recorder());
            blas.gemv(x, ones_rows, sums, SushiBLAS::Core::Transpose::Trans, 1.0f, 0.0f);
            context.count(1);
            return Status();
        }

        Status lower_gather_rows(const LoweringContext& context)
        {
            context.require_arity(2, 1);
            require_int32_index(context);

            const Value& table_value = context.input_value(0);
            SA_FATAL_IF(table_value.shape.rank() != 2,
                        "GATHER_ROWS (node {}) needs a rank-2 table; got {}", context.id(),
                        table_value.shape.to_string());

            const SushiBLAS::Tensor& table = context.input(0);
            const SushiBLAS::Tensor& indices = context.input(1);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& gathered, context.output(0));
            SA_TRY_ASSIGN(const SushiBLAS::SparseMatrix selection,
                          row_selection(context, indices, table_value.shape[0],
                                        table_value.dtype));

            SushiBLAS::Tensor rows = as_rows(context, gathered, selection.rows,
                                             table_value.shape[1]);

            // Each row of the selection holds a single one, so the product copies a table row.
            SushiBLAS::SparseOps sparse(context.recorder());
            sparse.spmm(selection, table, rows, SushiBLAS::Core::Transpose::NoTrans, 1.0, 0.0);
            context.count(1);
            return Status();
        }

        Status lower_scatter_add_rows(const LoweringContext& context)
        {
            context.require_arity(2, 1);
            require_int32_index(context);

            const Value& sums_value = context.output_value(0);
            SA_FATAL_IF(sums_value.shape.rank() != 2,
                        "SCATTER_ADD_ROWS (node {}) needs a rank-2 result; got {}", context.id(),
                        sums_value.shape.to_string());

            const SushiBLAS::Tensor& source = context.input(0);
            const SushiBLAS::Tensor& indices = context.input(1);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& sums, context.output(0));
            SA_TRY_ASSIGN(const SushiBLAS::SparseMatrix selection,
                          row_selection(context, indices, sums_value.shape[0],
                                        sums_value.dtype));

            const SushiBLAS::Tensor rows = as_rows(context, source, selection.rows,
                                                   sums_value.shape[1]);

            // The transposed product adds the rows in index order, one work-item per column.
            SushiBLAS::SparseOps sparse(context.recorder());
            sparse.spmm(selection, rows, sums, SushiBLAS::Core::Transpose::Trans, 1.0, 0.0);
            context.count(1);
            return Status();
        }

        Status lower_causal_mask(const LoweringContext& context)
        {
            context.require_arity(1, 1);

            const Value& scores_value = context.input_value(0);
            const Shape& shape = scores_value.shape;
            const std::int32_t rank = shape.rank();
            SA_FATAL_IF(rank < 2 || shape[rank - 1] != shape[rank - 2] || shape[rank - 1] < 1,
                        "CAUSAL_MASK (node {}) needs [..., T, T] scores; got {}", context.id(),
                        shape.to_string());

            const std::int64_t length = shape[rank - 1];
            const std::int64_t entries = length * length;
            const std::int64_t matrices = shape.num_elements() / entries;

            const SushiBLAS::Tensor& scores = context.input(0);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& masked, context.output(0));
            SA_TRY_ASSIGN(const TriangleAffine affine,
                          triangle_affine(context, length, context.node().attributes.get(0),
                                          scores_value.dtype));

            const SushiBLAS::Tensor source = as_entry_channels(context, scores, matrices, entries);
            SushiBLAS::Tensor target = as_entry_channels(context, masked, matrices, entries);

            // A kept entry is scaled by one and shifted by zero; a masked one by zero and the fill.
            SushiBLAS::SpatialOps spatial(context.recorder());
            spatial.channel_affine(source, *affine.keep, *affine.shift, target);
            context.count(1);
            return Status();
        }
    } // namespace Graph
} // namespace SushiAI
