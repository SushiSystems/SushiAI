/**************************************************************************/
/* darknet_weights.cpp                                                    */
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
 * @file darknet_weights.cpp
 * @brief Implements the .weights header reader and the strict, layer-by-layer float reader.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/importers/darknet_weights.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <istream>
#include <limits>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

static_assert(std::numeric_limits<float>::is_iec559 && sizeof(float) == 4,
              "the .weights reader stores each float as an IEEE 754 binary32 bit pattern");

namespace SushiAI
{
    namespace Vision::Importers
    {
        namespace
        {
            /** @brief Holds the largest major or minor version this reader accepts. */
            constexpr std::int64_t K_MAX_VERSION_PART = 1000;

            /** @brief Holds the smallest major * 10 + minor whose seen count is 64 bits wide. */
            constexpr std::int64_t K_WIDE_SEEN_FROM = 2;

            /** @brief Holds the bytes of the three int32 version fields. */
            constexpr std::uint64_t K_VERSION_BYTES = 12;

            /** @brief Holds the bytes of one float32. */
            constexpr std::uint64_t K_FLOAT_BYTES = 4;

            /** @brief Holds the largest value a std::uint64_t can hold. */
            constexpr std::uint64_t K_MAX_UNSIGNED = std::numeric_limits<std::uint64_t>::max();

            /**
             * @brief Returns the width in bytes of the seen count that a file's version implies.
             * @return 8 from version 0.2 on, 4 below it.
             */
            [[nodiscard]] std::size_t seen_width(std::int32_t major, std::int32_t minor) noexcept
            {
                // The product is taken in 64 bits, so no pair of int32 fields overflows it.
                const std::int64_t version = static_cast<std::int64_t>(major) * 10 + minor;
                return version >= K_WIDE_SEEN_FROM ? 8 : 4;
            }

            /**
             * @brief Assembles little-endian bytes into one unsigned value.
             * @param bytes At least @p width readable bytes, least significant first.
             * @param width The number of bytes, from 1 to 8.
             * @return The value the bytes spell.
             */
            [[nodiscard]] std::uint64_t decode_little_endian(const unsigned char* bytes,
                                                             std::size_t width) noexcept
            {
                std::uint64_t value = 0;
                for (std::size_t i = 0; i < width; ++i)
                    value |= static_cast<std::uint64_t>(bytes[i]) << (8 * i);
                return value;
            }

            /**
             * @brief Reads one little-endian header field of @p width bytes.
             * @param what Names the field in the failure's message.
             * @return The value, or InvalidFile when the stream ends first.
             */
            [[nodiscard]] Result<std::uint64_t> read_field(std::istream& stream, std::size_t width,
                                                           std::string_view what)
            {
                unsigned char bytes[8] = {};
                stream.read(reinterpret_cast<char*>(bytes), static_cast<std::streamsize>(width));
                SA_FAIL_IF(!stream, StatusCode::InvalidFile,
                           "the .weights header ends before {}", what);
                return decode_little_endian(bytes, width);
            }

            /**
             * @brief Reads one little-endian int32 header field.
             * @param what Names the field in the failure's message.
             * @return The value, or InvalidFile when the stream ends first.
             */
            [[nodiscard]] Result<std::int32_t> read_int32(std::istream& stream,
                                                          std::string_view what)
            {
                SA_TRY_ASSIGN(const std::uint64_t bits, read_field(stream, 4, what));
                return static_cast<std::int32_t>(static_cast<std::uint32_t>(bits));
            }

            /**
             * @brief Reads @p count float32 values from the stream into a new vector.
             * @param buffer A byte buffer the caller keeps, so the blocks of one file reuse it.
             * @param layer The convolution's layer index, named in a failure.
             * @param field Which vector the values fill, named in a failure.
             * @return The values, or InvalidFile when the stream ends first.
             */
            [[nodiscard]] Result<std::vector<float>> read_floats(std::istream& stream,
                                                                 std::uint64_t count,
                                                                 std::vector<unsigned char>& buffer,
                                                                 std::int64_t layer,
                                                                 std::string_view field)
            {
                const std::size_t byte_count = static_cast<std::size_t>(count * K_FLOAT_BYTES);
                buffer.resize(byte_count);
                stream.read(reinterpret_cast<char*>(buffer.data()),
                            static_cast<std::streamsize>(byte_count));
                SA_FAIL_IF(!stream, StatusCode::InvalidFile,
                           "the .weights file ends inside convolution layer {}'s {}", layer, field);

                std::vector<float> values(static_cast<std::size_t>(count));
                for (std::size_t i = 0; i < values.size(); ++i)
                {
                    const unsigned char* bytes = buffer.data() + i * K_FLOAT_BYTES;
                    const auto bits = static_cast<std::uint32_t>(
                        decode_little_endian(bytes, static_cast<std::size_t>(K_FLOAT_BYTES)));
                    std::memcpy(&values[i], &bits, sizeof bits);
                }
                return values;
            }

            /**
             * @brief Multiplies two counts, or reports that the product does not fit.
             * @param product Receives the product when it fits; left unchanged otherwise.
             * @return False when the product would overflow std::uint64_t.
             */
            [[nodiscard]] bool checked_multiply(std::uint64_t a, std::uint64_t b,
                                                std::uint64_t& product) noexcept
            {
                if (a != 0 && b > K_MAX_UNSIGNED / a)
                    return false;

                product = a * b;
                return true;
            }

            /**
             * @brief Adds two counts, or reports that the sum does not fit.
             * @param sum Receives the sum when it fits; left unchanged otherwise.
             * @return False when the sum would overflow std::uint64_t.
             */
            [[nodiscard]] bool checked_add(std::uint64_t a, std::uint64_t b,
                                           std::uint64_t& sum) noexcept
            {
                if (b > K_MAX_UNSIGNED - a)
                    return false;

                sum = a + b;
                return true;
            }

            /**
             * @brief Counts the floats of a layer's weights, which are [out, in, kh, kw].
             * @param shape The layer; a field that is not positive is InvalidDescription.
             * @return The count, or InvalidDescription when the layer is too large to count.
             */
            [[nodiscard]] Result<std::uint64_t> weight_count(const DarknetConvolutionShape& shape)
            {
                SA_FAIL_IF(shape.filters <= 0 || shape.channels <= 0 || shape.size <= 0,
                           StatusCode::InvalidDescription,
                           "convolution layer {} has filters {}, channels {} and size {}; "
                           "each must be positive",
                           shape.layer, shape.filters, shape.channels, shape.size);

                const auto filters = static_cast<std::uint64_t>(shape.filters);
                const auto channels = static_cast<std::uint64_t>(shape.channels);
                const auto size = static_cast<std::uint64_t>(shape.size);

                std::uint64_t kernel_area = 0;
                std::uint64_t per_filter = 0;
                std::uint64_t count = 0;
                SA_FAIL_IF(!checked_multiply(size, size, kernel_area)
                               || !checked_multiply(channels, kernel_area, per_filter)
                               || !checked_multiply(filters, per_filter, count),
                           StatusCode::InvalidDescription,
                           "convolution layer {} is too large to read", shape.layer);
                return count;
            }

            /**
             * @brief Counts the bytes a layer occupies: its per-filter vectors, then its weights.
             * @param shape The layer, validated by weight_count().
             * @return The byte count, or InvalidDescription when it does not fit in 64 bits.
             */
            [[nodiscard]] Result<std::uint64_t> layer_bytes(const DarknetConvolutionShape& shape)
            {
                SA_TRY_ASSIGN(const std::uint64_t weights, weight_count(shape));

                const std::uint64_t filters = static_cast<std::uint64_t>(shape.filters);
                const std::uint64_t vectors_per_filter = shape.normalised ? 4 : 1;
                std::uint64_t filter_floats = 0;
                std::uint64_t floats = 0;
                std::uint64_t bytes = 0;
                SA_FAIL_IF(!checked_multiply(filters, vectors_per_filter, filter_floats)
                               || !checked_add(filter_floats, weights, floats)
                               || !checked_multiply(floats, K_FLOAT_BYTES, bytes),
                           StatusCode::InvalidDescription,
                           "convolution layer {} is too large to read", shape.layer);
                return bytes;
            }

            /**
             * @brief Checks that the file holds exactly the bytes its header and layers need.
             * @param path Named in every failure.
             * @param header_bytes The version fields and the seen count.
             * @param file_bytes The size of the file.
             * @return Success, or InvalidDescription for a bad shape, or InvalidFile naming the
             *         layer where the file ends early, or the layer after which bytes are left.
             */
            [[nodiscard]] Status check_layout(const std::filesystem::path& path,
                                              SushiRuntime::span<const DarknetConvolutionShape>
                                                  shapes,
                                              std::uint64_t header_bytes, std::uint64_t file_bytes)
            {
                std::uint64_t required = header_bytes;
                for (const DarknetConvolutionShape& shape : shapes)
                {
                    SA_TRY_ASSIGN(const std::uint64_t bytes, layer_bytes(shape));

                    std::uint64_t end = 0;
                    SA_FAIL_IF(!checked_add(required, bytes, end), StatusCode::InvalidDescription,
                               "convolution layer {} is too large to read", shape.layer);
                    required = end;

                    SA_FAIL_IF(required > file_bytes, StatusCode::InvalidFile,
                               "'{}' is {} bytes, but convolution layer {} needs bytes up to {}; "
                               "the file ends early",
                               path.string(), file_bytes, shape.layer, required);
                }

                SA_FAIL_IF(required < file_bytes && shapes.size() == 0, StatusCode::InvalidFile,
                           "'{}' is {} bytes, but no convolution is listed, so {} bytes are left "
                           "over after the header",
                           path.string(), file_bytes, file_bytes - required);

                SA_FAIL_IF(required < file_bytes, StatusCode::InvalidFile,
                           "'{}' is {} bytes, but its layers need {} bytes: {} bytes are left "
                           "over after convolution layer {}",
                           path.string(), file_bytes, required, file_bytes - required,
                           shapes[shapes.size() - 1].layer);

                return Status();
            }

        } // namespace

        Result<DarknetWeightsHeader> read_darknet_header(std::istream& stream)
        {
            DarknetWeightsHeader header;
            SA_TRY_ASSIGN(header.major, read_int32(stream, "the major version"));
            SA_TRY_ASSIGN(header.minor, read_int32(stream, "the minor version"));

            // The seen count's width depends on the version, so the version is checked first.
            SA_FAIL_IF(header.major > K_MAX_VERSION_PART || header.minor > K_MAX_VERSION_PART,
                       StatusCode::Unsupported,
                       "the file is version {}.{}; this reader accepts a major and a minor up "
                       "to {}",
                       header.major, header.minor, K_MAX_VERSION_PART);

            SA_TRY_ASSIGN(header.revision, read_int32(stream, "the revision"));
            SA_TRY_ASSIGN(header.seen, read_field(stream, seen_width(header.major, header.minor),
                                                  "the seen count"));
            return header;
        }

        Result<DarknetWeights> read_darknet_weights(
            const std::filesystem::path& path,
            SushiRuntime::span<const DarknetConvolutionShape> shapes)
        {
            std::ifstream stream(path, std::ios::binary);
            SA_FAIL_IF(!stream, StatusCode::FileNotFound, "could not open '{}' for reading",
                       path.string());

            std::error_code error;
            const std::uintmax_t file_size = std::filesystem::file_size(path, error);
            SA_FAIL_IF(error, StatusCode::InvalidFile, "could not read the size of '{}': {}",
                       path.string(), error.message());

            const Result<DarknetWeightsHeader> header = read_darknet_header(stream);
            if (!header.has_value())
                return SA_ANNOTATE(header.status(), "'{}'", path.string());

            const DarknetWeightsHeader& fields = header.value();
            const auto seen = static_cast<std::uint64_t>(seen_width(fields.major, fields.minor));
            const std::uint64_t header_bytes = K_VERSION_BYTES + seen;
            SA_TRY(check_layout(path, shapes, header_bytes, static_cast<std::uint64_t>(file_size)));

            DarknetWeights weights;
            weights.header = fields;
            weights.convolutions.reserve(shapes.size());

            std::vector<unsigned char> buffer;
            for (const DarknetConvolutionShape& shape : shapes)
            {
                const auto filters = static_cast<std::uint64_t>(shape.filters);
                SA_TRY_ASSIGN(const std::uint64_t weight_floats, weight_count(shape));

                DarknetConvolution convolution;
                SA_TRY_ASSIGN(convolution.biases,
                              read_floats(stream, filters, buffer, shape.layer, "biases"));
                if (shape.normalised)
                {
                    SA_TRY_ASSIGN(convolution.scales,
                                  read_floats(stream, filters, buffer, shape.layer, "scales"));
                    SA_TRY_ASSIGN(convolution.rolling_mean,
                                  read_floats(stream, filters, buffer, shape.layer,
                                              "rolling means"));
                    SA_TRY_ASSIGN(convolution.rolling_variance,
                                  read_floats(stream, filters, buffer, shape.layer,
                                              "rolling variances"));
                }
                SA_TRY_ASSIGN(convolution.weights,
                              read_floats(stream, weight_floats, buffer, shape.layer, "weights"));

                weights.convolutions.push_back(std::move(convolution));
            }
            return weights;
        }

    } // namespace Vision::Importers
} // namespace SushiAI
