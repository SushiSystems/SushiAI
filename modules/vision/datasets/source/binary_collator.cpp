/**************************************************************************/
/* binary_collator.cpp                                                    */
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
 * @file binary_collator.cpp
 * @brief Implements the collator of the binary task.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/datasets/binary_collator.hpp>

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
        constexpr std::string_view K_COLLATOR = "BinaryCollator";
    } // namespace

    Result<std::unique_ptr<BinaryCollator>> BinaryCollator::make(std::int32_t positive_class)
    {
        SA_FAIL_IF(positive_class < 0 || positive_class >= K_BINARY_CLASSES,
                   StatusCode::InvalidDescription,
                   "{}: the positive class is {}; a set of {} classes has 0 and 1", K_COLLATOR,
                   positive_class, K_BINARY_CLASSES);

        return Detail::at_boundary(K_COLLATOR, [&]() -> Result<std::unique_ptr<BinaryCollator>>
        {
            return std::unique_ptr<BinaryCollator>(new BinaryCollator(positive_class));
        });
    }

    BinaryCollator::BinaryCollator(std::int32_t positive_class)
        : positive_class_(positive_class),
          targets_{Data::TargetDescription{std::string(Data::K_ROLE_LABELS), Shape{},
                                           DType::Float32, 0}}
    {

    }

    std::string_view BinaryCollator::task() const noexcept
    {
        return K_TASK_BINARY;
    }

    const std::vector<Data::TargetDescription>& BinaryCollator::targets() const noexcept
    {
        return targets_;
    }

    Status BinaryCollator::collate(const SampleList& list,
                                   SushiRuntime::span<const FittedSample> samples,
                                   Data::Batch& batch) const
    {
        SA_TRY(write_batch_inputs(samples, batch.inputs));
        SA_TRY_ASSIGN(SushiBLAS::Tensor* const labels,
                      Detail::role_tensor(batch, Data::K_ROLE_LABELS, DType::Float32,
                                          static_cast<std::int64_t>(samples.size()),
                                          K_COLLATOR));

        float* out = labels->data_as<float>();

        for (std::size_t i = 0; i < samples.size(); ++i)
        {
            const FittedSample& sample = samples[i];

            SA_FAIL_IF(!sample.class_index.has_value(), StatusCode::InvalidDescription,
                       "'{}': {}: the sample has no class index", Detail::file_of(list, sample),
                       K_COLLATOR);
            SA_FAIL_IF(*sample.class_index < 0 || *sample.class_index >= K_BINARY_CLASSES,
                       StatusCode::InvalidDescription,
                       "'{}': {}: the class index {} is outside [0, {})",
                       Detail::file_of(list, sample), K_COLLATOR, *sample.class_index,
                       K_BINARY_CLASSES);

            out[i] = *sample.class_index == positive_class_ ? 1.0f : 0.0f;
        }

        return Status();
    }
} // namespace SushiAI::Vision::Datasets
