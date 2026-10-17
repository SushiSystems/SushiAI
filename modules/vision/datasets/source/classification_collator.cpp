/**************************************************************************/
/* classification_collator.cpp                                            */
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
 * @file classification_collator.cpp
 * @brief Implements the collator of the classification task.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/datasets/classification_collator.hpp>

#include <cstddef>
#include <string>

#include <SushiBLAS/tensor/tensor.hpp>

#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/vision/datasets/batch_inputs.hpp>

#include "collate_support.hpp"
#include "disk.hpp"

namespace SushiAI::Vision::Datasets
{
    namespace
    {
        /** @brief Names this unit in a message. */
        constexpr std::string_view K_COLLATOR = "ClassificationCollator";
    } // namespace

    Result<std::unique_ptr<ClassificationCollator>> ClassificationCollator::make(
        std::int32_t class_count)
    {
        SA_FAIL_IF(class_count < 1, StatusCode::InvalidDescription,
                   "{}: {} classes; there must be at least 1", K_COLLATOR, class_count);

        return Detail::at_boundary(
            K_COLLATOR, [&]() -> Result<std::unique_ptr<ClassificationCollator>>
        {
            return std::unique_ptr<ClassificationCollator>(
                new ClassificationCollator(class_count));
        });
    }

    ClassificationCollator::ClassificationCollator(std::int32_t class_count)
        : class_count_(class_count),
          targets_{Data::TargetDescription{std::string(Data::K_ROLE_LABELS), Shape{},
                                           DType::Int32, class_count}}
    {

    }

    std::string_view ClassificationCollator::task() const noexcept
    {
        return K_TASK_CLASSIFICATION;
    }

    const std::vector<Data::TargetDescription>& ClassificationCollator::targets() const noexcept
    {
        return targets_;
    }

    Status ClassificationCollator::collate(const SampleList& list,
                                           SushiRuntime::span<const FittedSample> samples,
                                           Data::Batch& batch) const
    {
        SA_TRY(write_batch_inputs(samples, batch.inputs));
        SA_TRY_ASSIGN(SushiBLAS::Tensor* const labels,
                      Detail::role_tensor(batch, Data::K_ROLE_LABELS, DType::Int32,
                                          static_cast<std::int64_t>(samples.size()),
                                          K_COLLATOR));

        std::int32_t* out = labels->data_as<std::int32_t>();

        for (std::size_t i = 0; i < samples.size(); ++i)
        {
            const FittedSample& sample = samples[i];

            SA_FAIL_IF(!sample.class_index.has_value(), StatusCode::InvalidDescription,
                       "'{}': {}: the sample has no class index", Detail::file_of(list, sample),
                       K_COLLATOR);
            SA_FAIL_IF(*sample.class_index < 0 || *sample.class_index >= class_count_,
                       StatusCode::InvalidDescription,
                       "'{}': {}: the class index {} is outside [0, {})",
                       Detail::file_of(list, sample), K_COLLATOR, *sample.class_index,
                       class_count_);

            out[i] = *sample.class_index;
        }

        return Status();
    }
} // namespace SushiAI::Vision::Datasets
