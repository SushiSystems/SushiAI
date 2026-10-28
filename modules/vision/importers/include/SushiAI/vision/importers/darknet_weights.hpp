/**************************************************************************/
/* darknet_weights.hpp                                                    */
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
 * @file darknet_weights.hpp
 * @brief Declares the reader of a Darknet .weights file and the tensors it yields.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        /**
         * @brief Holds the version fields and the seen count that open a .weights file.
         */
        struct DarknetWeightsHeader
        {
            /** @brief Holds the major version, the file's first int32. */
            std::int32_t major = 0;

            /** @brief Holds the minor version, the file's second int32. */
            std::int32_t minor = 0;

            /** @brief Holds the revision, the file's third int32. */
            std::int32_t revision = 0;

            /** @brief Holds the seen count; a 32-bit count on disk is widened to 64 bits. */
            std::uint64_t seen = 0;
        };

        /**
         * @brief Describes one convolutional layer, so the reader knows how many floats it holds.
         *
         * The reader takes these from the caller; it does not learn them from the file.
         */
        struct DarknetConvolutionShape
        {
            /** @brief Holds the layer's index in the .cfg, which a failure names. */
            std::int64_t layer = 0;

            /** @brief Holds the number of output filters, which is the bias count. */
            std::int64_t filters = 0;

            /** @brief Holds the number of input channels per filter. */
            std::int64_t channels = 0;

            /** @brief Holds the side of the square kernel. */
            std::int64_t size = 0;

            /** @brief Holds whether the layer carries the three batch-norm vectors. */
            bool normalised = false;
        };

        /**
         * @brief Holds the floats of one convolutional layer, in the order the file stores them.
         */
        struct DarknetConvolution
        {
            /** @brief Holds one bias per filter. */
            std::vector<float> biases;

            /** @brief Holds one scale per filter when normalised; empty otherwise. */
            std::vector<float> scales;

            /** @brief Holds one rolling mean per filter when normalised; empty otherwise. */
            std::vector<float> rolling_mean;

            /** @brief Holds one rolling variance per filter when normalised; empty otherwise. */
            std::vector<float> rolling_variance;

            /** @brief Holds the weights as [out, in, kh, kw], exactly as the file orders them. */
            std::vector<float> weights;
        };

        /**
         * @brief Holds a whole .weights file: its header and one entry per convolution.
         */
        struct DarknetWeights
        {
            /** @brief Holds the header the file opens with. */
            DarknetWeightsHeader header;

            /** @brief Holds one entry per convolutional layer, in file order. */
            std::vector<DarknetConvolution> convolutions;
        };

        /**
         * @brief Reads the version fields and the seen count from the start of @p stream.
         * @param stream Positioned at the file's first byte; it is left after the header.
         * @return The header; Unsupported when major or minor is above 1000, or InvalidFile when
         *         the stream ends inside the header.
         */
        [[nodiscard]] Result<DarknetWeightsHeader> read_darknet_header(std::istream& stream) noexcept;

        /**
         * @brief Reads every convolution's floats from a .weights file, in the order of @p shapes.
         * @param path The file; it must hold exactly the header and the floats the shapes name.
         * @param shapes One entry per convolutional layer, in file order.
         * @return The tensors; FileNotFound when the file cannot be opened; InvalidDescription for
         *         a shape that is not positive or too large to count; InvalidFile when the file is
         *         shorter or longer than its layers need, naming the layer and both byte counts.
         */
        [[nodiscard]] Result<DarknetWeights> read_darknet_weights(
            const std::filesystem::path& path,
            SushiRuntime::span<const DarknetConvolutionShape> shapes) noexcept;

    } // namespace Vision::Importers
} // namespace SushiAI
