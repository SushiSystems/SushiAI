/**************************************************************************/
/* regression_collator.cpp                                                */
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
 * @file regression_collator.cpp
 * @brief Implements the collator of the regression task over a tabular list.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/datasets/regression_collator.hpp>

#include <algorithm>
#include <cstddef>
#include <string>

#include <SushiBLAS/core/format.hpp>
#include <SushiBLAS/tensor/tensor.hpp>

#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/data/roles.hpp>

#include "collate_support.hpp"
#include "disk.hpp"

namespace SushiAI::Vision::Datasets
{
    namespace
    {
        /** @brief Names this unit in a message. */
        constexpr std::string_view K_COLLATOR = "RegressionCollator";

        /** @brief Holds the rank of the inputs: the batch axis and the feature axis. */
        constexpr std::int32_t K_INPUT_RANK = 2;
    } // namespace

    Result<std::unique_ptr<RegressionCollator>> RegressionCollator::make(
        std::int32_t feature_count, std::int32_t target_count)
    {
        SA_FAIL_IF(feature_count < 1, StatusCode::InvalidDescription,
                   "{}: {} features; there must be at least 1", K_COLLATOR, feature_count);
        SA_FAIL_IF(target_count < 1, StatusCode::InvalidDescription,
                   "{}: {} targets; there must be at least 1", K_COLLATOR, target_count);

        return Detail::at_boundary(
            K_COLLATOR, [&]() -> Result<std::unique_ptr<RegressionCollator>>
        {
            return std::unique_ptr<RegressionCollator>(
                new RegressionCollator(feature_count, target_count));
        });
    }

    RegressionCollator::RegressionCollator(std::int32_t feature_count, std::int32_t target_count)
        : feature_count_(feature_count), target_count_(target_count),
          targets_{Data::TargetDescription{std::string(Data::K_ROLE_VALUES), Shape{target_count},
                                           DType::Float32, 0}}
    {

    }

    std::string_view RegressionCollator::task() const noexcept
    {
        return K_TASK_REGRESSION;
    }

    const std::vector<Data::TargetDescription>& RegressionCollator::targets() const noexcept
    {
        return targets_;
    }

    Status RegressionCollator::collate(const SampleList& list,
                                       SushiRuntime::span<const FittedSample> samples,
                                       Data::Batch& batch) const
    {
        const std::size_t features = static_cast<std::size_t>(feature_count_);
        const std::size_t targets = static_cast<std::size_t>(target_count_);
        const std::int64_t rows = static_cast<std::int64_t>(samples.size());

        SA_FAIL_IF(batch.inputs.dtype != DType::Float32 || batch.inputs.rank != K_INPUT_RANK ||
                       batch.inputs.shape[0] != rows || batch.inputs.shape[1] != feature_count_,
                   StatusCode::Incompatible,
                   "{}: the input tensor has rank {} and {} elements; it must be float32 of "
                   "[{}, {}]",
                   K_COLLATOR, batch.inputs.rank, batch.inputs.num_elements, rows,
                   feature_count_);
        SA_TRY_ASSIGN(SushiBLAS::Tensor* const values,
                      Detail::role_tensor(batch, Data::K_ROLE_VALUES, DType::Float32,
                                          rows * target_count_, K_COLLATOR));

        float* input_out = batch.inputs.data_as<float>();
        float* value_out = values->data_as<float>();

        for (std::size_t i = 0; i < samples.size(); ++i)
        {
            const FittedSample& sample = samples[i];

            SA_FAIL_IF(sample.features.size() != features || sample.targets.size() != targets,
                       StatusCode::InvalidDescription,
                       "'{}': {}: the sample holds {} features and {} targets; the task reads "
                       "{} and {}",
                       Detail::file_of(list, sample), K_COLLATOR, sample.features.size(),
                       sample.targets.size(), features, targets);

            std::copy_n(sample.features.data(), features, input_out + i * features);
            std::copy_n(sample.targets.data(), targets, value_out + i * targets);
        }

        return Status();
    }
} // namespace SushiAI::Vision::Datasets
