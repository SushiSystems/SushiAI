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
 * @file batch_inputs.hpp
 * @brief Declares the copy of the fitted images of one batch into its input tensor.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/datasets/fitted_sample.hpp>

/** @brief Holds the readers of the on-disk layouts vision datasets are kept in. */
namespace SushiAI::Vision::Datasets
{
    /** @brief Holds the divisor that takes a byte to a float in [0, 1]. */
    inline constexpr float K_BYTE_SCALE = 255.0f;

    /**
     * @brief Writes each image, divided by K_BYTE_SCALE, into one row of an NHWC tensor.
     * @param inputs A float32 tensor of [N, H, W, C] in host-readable storage.
     * @return Incompatible when the tensor is not float32 of rank 4, when N is not the
     *         sample count, or when an image is not H by W of C channels; the message gives
     *         the position of the image in the batch.
     */
    [[nodiscard]] Status write_batch_inputs(SushiRuntime::span<const FittedSample> samples,
                                            SushiBLAS::Tensor& inputs) noexcept;
} // namespace SushiAI::Vision::Datasets
