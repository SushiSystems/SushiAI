/**************************************************************************/
/* batch_inputs.cpp                                                       */
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
 * @file batch_inputs.cpp
 * @brief Implements the copy of the fitted images of one batch into its input tensor.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/datasets/batch_inputs.hpp>

#include <cstddef>
#include <cstdint>
#include <string_view>


#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/vision/imaging/image.hpp>

namespace SushiAI::Vision::Datasets
{
    namespace
    {
        /** @brief Names this unit in a message. */
        constexpr std::string_view K_WRITER = "write_batch_inputs";

        /** @brief Holds the rank of an NHWC tensor. */
        constexpr std::int32_t K_INPUT_RANK = 4;
    } // namespace

    Status write_batch_inputs(SushiRuntime::span<const FittedSample> samples,
                              SushiBLAS::Tensor& inputs)
    {
        SA_FAIL_IF(inputs.dtype != DType::Float32 || inputs.rank != K_INPUT_RANK,
                   StatusCode::Incompatible,
                   "{}: the input tensor has rank {}; it must be float32 of rank {}", K_WRITER,
                   inputs.rank, K_INPUT_RANK);
        SA_FAIL_IF(inputs.shape[0] != static_cast<std::int64_t>(samples.size()),
                   StatusCode::Incompatible, "{}: the input tensor holds {} images; got {}",
                   K_WRITER, inputs.shape[0], samples.size());

        const std::int64_t height = inputs.shape[1];
        const std::int64_t width = inputs.shape[2];
        const std::int64_t channels = inputs.shape[3];
        const std::size_t elements = static_cast<std::size_t>(height * width * channels);
        float* out = inputs.data_as<float>();

        for (std::size_t i = 0; i < samples.size(); ++i)
        {
            const Imaging::Image& image = samples[i].image;

            SA_FAIL_IF(image.height() != height || image.width() != width ||
                           image.channels() != channels,
                       StatusCode::Incompatible,
                       "{}: image {} of the batch is {} by {} of {} channels; the input is {} "
                       "by {} of {}",
                       K_WRITER, i, image.height(), image.width(), image.channels(), height,
                       width, channels);

            const std::uint8_t* bytes = image.data();
            for (std::size_t e = 0; e < elements; ++e)
                out[e] = static_cast<float>(bytes[e]) / K_BYTE_SCALE;

            out += elements;
        }

        return Status();
    }
} // namespace SushiAI::Vision::Datasets
