/**************************************************************************/
/* batch_inputs.hpp                                                       */
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
 * @file batch_inputs.hpp
 * @brief Declares the conversion of the fitted images of one batch into its input tensor.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>

#include <SushiAI/config/input_normalise.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/datasets/fitted_sample.hpp>

/** @brief Holds the readers of the on-disk layouts vision datasets are kept in. */
namespace SushiAI::Vision::Datasets
{
    /**
     * @brief Writes each image into one row of an NHWC tensor as (level * scale - mean) / deviation.
     * @param inputs A float32 tensor of [N, H, W, C] in host-readable storage.
     * @param normalise The conversion; the default gives the byte divided by 255, bit for bit.
     * @return Incompatible when the tensor is not float32 of rank 4, when N is not the sample
     *         count, when an image is not H by W of C channels, or when C is not 3 and
     *         @p normalise states an order, a mean or a deviation; the message names the image.
     */
    [[nodiscard]] Status write_batch_inputs(
        SushiRuntime::span<const FittedSample> samples, SushiBLAS::Tensor& inputs,
        const Config::InputNormalise& normalise = Config::InputNormalise()) noexcept;
} // namespace SushiAI::Vision::Datasets
