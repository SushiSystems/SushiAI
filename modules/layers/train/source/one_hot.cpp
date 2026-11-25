/**************************************************************************/
/* one_hot.cpp                                                            */
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
 * @file one_hot.cpp
 * @brief Implements the host fill that turns a batch's class indices into one-hot rows.
 * @author Mustafa Garip
 */

#include <SushiAI/train/one_hot.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/allocation/host_copy.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/shape/tensor_shape.hpp>

namespace SushiAI
{
    namespace Train
    {
        namespace
        {
            /**
             * @brief Returns @p rows rows of @p classes values, one 1 per row and 0 elsewhere.
             * @pre Every label is in [0, classes).
             */
            [[nodiscard]] std::vector<double> one_hot_rows(const int32_t* labels, int64_t rows,
                                                           int64_t classes)
            {
                std::vector<double> values(static_cast<std::size_t>(rows * classes), 0.0);
                for (int64_t row = 0; row < rows; ++row)
                    values[static_cast<std::size_t>(row * classes + labels[row])] = 1.0;
                return values;
            }
        } // namespace

        Status fill_one_hot(const SushiBLAS::Tensor& labels, SushiBLAS::Tensor& one_hot) noexcept
        {
            SA_TRY_ASSIGN(const Shape label_shape, checked_shape_of(labels));
            SA_TRY_ASSIGN(const Shape row_shape, checked_shape_of(one_hot));

            SA_FAIL_IF(label_shape.rank() != 1 || row_shape.rank() != 2
                       || label_shape[0] != row_shape[0],
                       StatusCode::Incompatible,
                       "The batch's labels have shape {} but the compiled step expects one "
                       "label per row of its {} targets",
                       label_shape.to_string(), row_shape.to_string());
            SA_FAIL_IF(labels.dtype != DType::Int32, StatusCode::Incompatible,
                       "The batch's labels are {} but the compiled step reads {}",
                       dtype_name(labels.dtype), dtype_name(DType::Int32));
            SA_FAIL_IF(one_hot.dtype != DType::Float32 && one_hot.dtype != DType::Float64,
                       StatusCode::Incompatible,
                       "The compiled step's targets are {}; one-hot rows are written as {} "
                       "or {}",
                       dtype_name(one_hot.dtype), dtype_name(DType::Float32),
                       dtype_name(DType::Float64));

            const int64_t rows = row_shape[0];
            const int64_t classes = row_shape[1];
            const int32_t* indices = labels.data_as<int32_t>();

            for (int64_t row = 0; row < rows; ++row)
            {
                SA_FAIL_IF(indices[row] < 0 || indices[row] >= classes, StatusCode::Incompatible,
                           "The label of row {} is {} but the compiled step has {} classes",
                           row, indices[row], classes);
            }

            SA_TRY_ASSIGN(const std::vector<double> values,
                          Detail::guard_value(Detail::SeamPhase::Allocate, [&]()
            {
                return one_hot_rows(indices, rows, classes);
            }));

            return write_reals(one_hot,
                               SushiRuntime::span<const double>(values.data(), values.size()));
        }

    } // namespace Train
} // namespace SushiAI
