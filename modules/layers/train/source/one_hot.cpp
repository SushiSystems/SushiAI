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
 * @file one_hot.cpp
 * @brief Implements the host fill that turns a batch's class indices into one-hot rows.
 * @author Mustafa Garip
 */

#include <SushiAI/train/one_hot.hpp>

#include <algorithm>
#include <cstdint>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/shape/tensor_shape.hpp>

namespace SushiAI
{
    namespace Train
    {
        namespace
        {
            /**
             * @brief Zeroes @p rows rows of @p classes elements and sets one element per row.
             * @pre Every label is in [0, classes).
             */
            template<typename Real>
            void write_rows(const int32_t* labels, Real* one_hot, int64_t rows, int64_t classes)
            {
                std::fill(one_hot, one_hot + rows * classes, Real(0));

                for (int64_t row = 0; row < rows; ++row)
                    one_hot[row * classes + labels[row]] = Real(1);
            }
        } // namespace

        Status fill_one_hot(const SushiBLAS::Tensor& labels, SushiBLAS::Tensor& one_hot)
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

            if (one_hot.dtype == DType::Float64)
                write_rows(indices, one_hot.data_as<double>(), rows, classes);
            else
                write_rows(indices, one_hot.data_as<float>(), rows, classes);

            return Status();
        }

    } // namespace Train
} // namespace SushiAI
