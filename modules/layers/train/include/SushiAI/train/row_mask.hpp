/**************************************************************************/
/* row_mask.hpp                                                           */
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
 * @file row_mask.hpp
 * @brief Declares the row mask: the 0 or 1 a batch states per sample, to drop rows from a loss.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <string_view>
#include <SushiAI/config/value.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/train/kept_rows.hpp>

namespace SushiAI
{
    namespace Train
    {
        class ObjectiveLeaves;

        /** @brief Names the key of an objective section that turns the row mask on. */
        inline constexpr std::string_view K_TYPE_MASK_KEY = "type_mask";

        /** @brief Names the key of the "type_mask" object that states k, the mask's columns. */
        inline constexpr std::string_view K_TYPE_MASK_COLUMNS_KEY = "columns";

        /** @brief Names the key of the "type_mask" object that picks the column; 0 when absent. */
        inline constexpr std::string_view K_TYPE_MASK_COLUMN_KEY = "column";

        /** @brief Names the leaf that holds the weight of every element of the loss. */
        inline constexpr std::string_view K_TYPE_MASK_WEIGHTS_LEAF = "type_mask_weights";

        /**
         * @brief Drops the rows a batch marks with 0 from a loss, and takes the mean over the rest.
         *
         * The key "type_mask" turns it on: true reads Data::K_ROLE_TYPE_MASK as [N];
         * {"columns": k, "column": c} reads [N, k] and uses column c. feed() weights a kept row
         * 1 over the kept count and a dropped row 0; a batch that drops every row has loss and
         * gradient 0. The default is off.
         */
        class RowMask final
        {
            public:
                /** @brief Builds the off mask: declare() and feed() do nothing. */
                RowMask() noexcept = default;

                /**
                 * @brief Builds a mask over column @p column of a role of @p columns columns.
                 * @param columns 0 reads the role as [N]; k >= 1 reads it as [N, k].
                 * @param column Picks the column; 0 when @p columns is 0.
                 * @return The mask, or InvalidDescription for a negative extent or a column
                 *         the role has none of.
                 */
                [[nodiscard]] static Result<RowMask> make(std::int64_t columns,
                                                          std::int64_t column) noexcept;

                /**
                 * @brief Reads the optional K_TYPE_MASK_KEY of an objective section.
                 * @param attributes The keys of the objective.
                 * @param context What to call the objective in a diagnostic.
                 * @return The mask; off when the key is absent or false; InvalidDescription
                 *         for a value that is neither a boolean nor an object, for another
                 *         key of the object, for a missing "columns" or for what make() refuses.
                 */
                [[nodiscard]] static Result<RowMask> read(const Config::Value& attributes,
                                                          std::string_view context) noexcept;

                /** @brief Returns true when the mask is on. */
                [[nodiscard]] bool is_active() const noexcept { return active_; }

                /** @brief Returns the columns of the role; 0 for a role of one value per sample. */
                [[nodiscard]] std::int64_t columns() const noexcept { return columns_; }

                /** @brief Returns the column the mask reads. */
                [[nodiscard]] std::int64_t column() const noexcept { return column_; }

                /**
                 * @brief Declares the weights leaf and requires the role; does nothing when off.
                 * @param element_shape The shape of the loss terms, [N, width].
                 * @param dtype The element type of the loss terms.
                 */
                void declare(const Graph::GraphBuilder& builder, ObjectiveLeaves& leaves,
                             const Shape& element_shape, DType dtype) const;

                /**
                 * @brief Reads which rows of @p batch the mask keeps.
                 * @return The rows, one per sample of the mask role; the selection that keeps
                 *         every row when the mask is off; InvalidDescription for a batch with
                 *         no mask role; or Incompatible for a mask that is not float32, holds
                 *         no row or a value that is neither 0 nor 1.
                 */
                [[nodiscard]] Result<KeptRows> kept_rows(const Data::Batch& batch) const noexcept;

                /**
                 * @brief Reads which of the @p rows rows of @p batch the mask keeps.
                 * @return What kept_rows() returns, or Incompatible when the mask role holds
                 *         another count of rows than @p rows; an off mask keeps every row.
                 */
                [[nodiscard]] Result<KeptRows> kept_rows_of(const Data::Batch& batch,
                                                            std::int64_t rows) const noexcept;

                /**
                 * @brief Writes the weights of @p batch into the leaf; does nothing when off.
                 * @return Success, InvalidDescription for a batch with no mask role, or
                 *         Incompatible for a mask that is not float32, holds another count
                 *         of values than the leaf's rows or a value that is neither 0 nor 1.
                 *         A refused batch leaves the leaf as it was.
                 */
                [[nodiscard]] Status feed(const Data::Batch& batch, const ObjectiveLeaves& leaves,
                                          Graph::TensorPool& pool) const;

                /**
                 * @brief Emits row_scale times the sum of every term times its weight.
                 * @param terms The [N, width] loss terms; they match the leaf declare() made.
                 * @param row_scale 1 / width turns a row of terms into its mean, 1 keeps its sum.
                 * @pre The mask is on and @p leaves holds the leaf declare() recorded.
                 * @return A 1-element value, or the failure of an emitter.
                 */
                [[nodiscard]] Graph::Traced emit_weighted(const Graph::GraphBuilder& builder,
                                                          const Graph::Graph& graph,
                                                          const Graph::Traced& terms,
                                                          const ObjectiveLeaves& leaves,
                                                          double row_scale) const noexcept;

            private:
                /** @brief Stores what make() has checked. */
                RowMask(std::int64_t columns, std::int64_t column) noexcept;

                /// @brief Whether the mask is on.
                bool active_ = false;

                /// @brief The columns of the role; 0 for [N].
                std::int64_t columns_ = 0;

                /// @brief The column the mask reads.
                std::int64_t column_ = 0;
        };

    } // namespace Train
} // namespace SushiAI
