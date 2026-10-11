/**************************************************************************/
/* dense_targets.cpp                                                      */
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
 * @file dense_targets.cpp
 * @brief Implements the host fills that copy the real values of a batch role into a target leaf.
 * @author Mustafa Garip
 */

#include <SushiAI/train/dense_targets.hpp>

#include <cstdint>
#include <SushiAI/core/common.hpp>

namespace SushiAI
{
    namespace Train
    {
        namespace
        {
            /** @brief Returns Incompatible unless @p values can be copied into @p targets. */
            [[nodiscard]] Status require_copyable(const SushiBLAS::Tensor& values,
                                                  const SushiBLAS::Tensor& targets)
            {
                SA_FAIL_IF(values.dtype != DType::Float32, StatusCode::Incompatible,
                           "The batch's values are {} but the compiled step reads {}",
                           dtype_name(values.dtype), dtype_name(DType::Float32));
                SA_FAIL_IF(targets.dtype != DType::Float32 && targets.dtype != DType::Float64,
                           StatusCode::Incompatible,
                           "The compiled step's targets are {}; real values are written as {} "
                           "or {}",
                           dtype_name(targets.dtype), dtype_name(DType::Float32),
                           dtype_name(DType::Float64));
                SA_FAIL_IF(values.num_elements != targets.num_elements, StatusCode::Incompatible,
                           "The batch holds {} values but the compiled step's targets hold {}",
                           values.num_elements, targets.num_elements);

                return Status();
            }

            /** @brief Writes the @p count elements of @p from into @p to as Real. */
            template<typename Real>
            void copy_values(const float* from, Real* to, std::int64_t count) noexcept
            {
                for (std::int64_t index = 0; index < count; ++index)
                    to[index] = static_cast<Real>(from[index]);
            }

            /** @brief Copies @p values into @p targets. @pre require_copyable() accepted them. */
            void copy_checked(const SushiBLAS::Tensor& values, SushiBLAS::Tensor& targets) noexcept
            {
                if (targets.dtype == DType::Float64)
                    copy_values(values.data_as<float>(), targets.data_as<double>(),
                                values.num_elements);
                else
                    copy_values(values.data_as<float>(), targets.data_as<float>(),
                                values.num_elements);
            }
        } // namespace

        Status fill_dense_targets(const SushiBLAS::Tensor& values, SushiBLAS::Tensor& targets)
        {
            SA_TRY(require_copyable(values, targets));
            copy_checked(values, targets);

            return Status();
        }

        Status fill_binary_targets(const SushiBLAS::Tensor& labels, SushiBLAS::Tensor& targets)
        {
            SA_TRY(require_copyable(labels, targets));

            const float* const label = labels.data_as<float>();
            for (std::int64_t index = 0; index < labels.num_elements; ++index)
            {
                SA_FAIL_IF(label[index] != 0.0f && label[index] != 1.0f,
                           StatusCode::Incompatible,
                           "The label of element {} is {}; a binary label is 0 or 1", index,
                           label[index]);
            }

            // Every label was checked, so a refused batch leaves the leaf as it was.
            copy_checked(labels, targets);

            return Status();
        }

    } // namespace Train
} // namespace SushiAI
