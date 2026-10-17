/**************************************************************************/
/* collate_support.hpp                                                    */
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
 * @file collate_support.hpp
 * @brief Declares the lookups the collators share: a role's tensor and a sample's file name.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include <SushiBLAS/tensor/tensor.hpp>

#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/vision/datasets/fitted_sample.hpp>
#include <SushiAI/vision/datasets/sample.hpp>

/** @brief Holds the units of the datasets module that no caller names. */
namespace SushiAI::Vision::Datasets::Detail
{
    /**
     * @brief Returns the path a fitted sample's pixels came from, for a message.
     * @return The path with forward slashes, or "file <index>" when the list lacks the index.
     */
    [[nodiscard]] std::string file_of(const SampleList& list, const FittedSample& sample);

    /**
     * @brief Returns the tensor a batch holds under a role, for writing.
     * @param collator Names the calling collator in a message.
     * @return The tensor, never null; Incompatible when the batch lacks the role, or holds
     *         it with another element type or another element count.
     */
    [[nodiscard]] Result<SushiBLAS::Tensor*> role_tensor(Data::Batch& batch,
                                                         std::string_view role, DType dtype,
                                                         std::int64_t elements,
                                                         std::string_view collator);
} // namespace SushiAI::Vision::Datasets::Detail
