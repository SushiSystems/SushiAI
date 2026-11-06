/**************************************************************************/
/* idx_file.cpp                                                           */
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
 * @file idx_file.cpp
 * @brief Implements the readers of an IDX file's header and payload.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/datasets/idx_file.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <ios>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>


#include "disk.hpp"

namespace SushiAI::Vision::Datasets
{
    namespace
    {
        /** @brief Names the header reader in a message. */
        constexpr std::string_view K_HEADER_READER = "read_idx_header";

        /** @brief Names the payload reader in a message. */
        constexpr std::string_view K_PAYLOAD_READER = "read_idx_payload";

        /** @brief Holds the byte count of the zero bytes, the element code and the rank. */
        constexpr std::size_t K_LEAD_BYTES = 4;

        /** @brief Holds the byte count of one extent. */
        constexpr std::size_t K_EXTENT_BYTES = 4;

        /** @brief Lists every element type the format has. */
        constexpr IDXElement K_ELEMENTS[] = {
            IDXElement::UInt8, IDXElement::Int8,    IDXElement::Int16,
            IDXElement::Int32, IDXElement::Float32, IDXElement::Float64,
        };

        /** @brief Reads a header; may throw what an allocation throws. */
        [[nodiscard]] Result<IDXHeader> read_header_checked(const std::filesystem::path& file,
                                                            const ByteSource& source)
        {
            const std::string name = Detail::display_of(file);
            SA_TRY_ASSIGN(const std::uint64_t file_bytes,
                          source.size_of(file, K_HEADER_READER));
            SA_TRY_ASSIGN(const std::string lead_text,
                          source.read_range(file, 0, K_LEAD_BYTES, K_HEADER_READER));

            SA_FAIL_IF(lead_text.size() != K_LEAD_BYTES, StatusCode::InvalidFile,
                       "'{}': {}: the file ends inside its {} leading bytes", name,
                       K_HEADER_READER, K_LEAD_BYTES);

            unsigned char lead[K_LEAD_BYTES] = {0, 0, 0, 0};
            std::copy(lead_text.begin(), lead_text.end(), lead);
            SA_FAIL_IF(lead[0] != 0 || lead[1] != 0, StatusCode::InvalidFile,
                       "'{}': {}: the first two bytes are {} and {}; an IDX file opens with two "
                       "zero bytes",
                       name, K_HEADER_READER, static_cast<int>(lead[0]),
                       static_cast<int>(lead[1]));

            IDXHeader header;
            bool is_known = false;
            for (IDXElement element : K_ELEMENTS)
            {
                if (static_cast<unsigned char>(element) == lead[2])
                {
                    header.element = element;
                    is_known = true;
                }
            }

            SA_FAIL_IF(!is_known, StatusCode::InvalidFile,
                       "'{}': {}: the element code {} is none of 8, 9, 11, 12, 13 and 14", name,
                       K_HEADER_READER, static_cast<int>(lead[2]));

            const std::size_t rank = lead[3];
            SA_FAIL_IF(rank == 0, StatusCode::InvalidFile,
                       "'{}': {}: the header declares no dimension", name, K_HEADER_READER);

            SA_TRY_ASSIGN(const std::string extent_text,
                          source.read_range(file, K_LEAD_BYTES, rank * K_EXTENT_BYTES,
                                            K_HEADER_READER));

            header.extents.reserve(rank);
            header.payload_bytes = idx_element_bytes(header.element);

            for (std::size_t dimension = 0; dimension < rank; ++dimension)
            {
                SA_FAIL_IF(extent_text.size() < (dimension + 1) * K_EXTENT_BYTES,
                           StatusCode::InvalidFile,
                           "'{}': {}: the file ends inside the extent of dimension {} of {}",
                           name, K_HEADER_READER, dimension, rank);

                unsigned char bytes[K_EXTENT_BYTES] = {0, 0, 0, 0};
                std::copy_n(extent_text.begin() + static_cast<std::ptrdiff_t>(
                                                      dimension * K_EXTENT_BYTES),
                            K_EXTENT_BYTES, bytes);

                const std::uint32_t extent = (static_cast<std::uint32_t>(bytes[0]) << 24) |
                                             (static_cast<std::uint32_t>(bytes[1]) << 16) |
                                             (static_cast<std::uint32_t>(bytes[2]) << 8) |
                                             static_cast<std::uint32_t>(bytes[3]);

                SA_FAIL_IF(extent != 0 && header.payload_bytes >
                                              std::numeric_limits<std::uint64_t>::max() / extent,
                           StatusCode::InvalidFile,
                           "'{}': {}: the extents up to dimension {} overflow 64 bits", name,
                           K_HEADER_READER, dimension);

                header.payload_bytes *= extent;
                header.extents.push_back(extent);
            }

            SA_FAIL_IF(file_bytes < header.header_bytes() ||
                           file_bytes - header.header_bytes() != header.payload_bytes,
                       StatusCode::InvalidFile,
                       "'{}': {}: the header declares {} bytes of elements after {} header "
                       "bytes; the file holds {} bytes",
                       name, K_HEADER_READER, header.payload_bytes, header.header_bytes(),
                       file_bytes);

            return Result<IDXHeader>(std::move(header));
        }

        /** @brief Reads a payload; may throw what an allocation throws. */
        [[nodiscard]] Result<std::vector<std::uint8_t>> read_payload_checked(
            const std::filesystem::path& file, const IDXHeader& header, const ByteSource& source)
        {
            const std::string name = Detail::display_of(file);
            SA_FAIL_IF(header.payload_bytes >
                           static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()),
                       StatusCode::Unsupported, "'{}': {}: {} bytes are more than one read holds",
                       name, K_PAYLOAD_READER, header.payload_bytes);

            SA_TRY_ASSIGN(const std::string text,
                          source.read_range(file, header.header_bytes(), header.payload_bytes,
                                            K_PAYLOAD_READER));
            SA_FAIL_IF(text.size() != header.payload_bytes, StatusCode::InvalidFile,
                       "'{}': {}: the read ended after {} of {} bytes", name, K_PAYLOAD_READER,
                       text.size(), header.payload_bytes);

            return Result<std::vector<std::uint8_t>>(
                std::vector<std::uint8_t>(text.begin(), text.end()));
        }
    } // namespace

    Result<IDXHeader> read_idx_header(const std::filesystem::path& file) noexcept
    {
        return read_idx_header(file, Detail::FileSystemSource());
    }

    Result<IDXHeader> read_idx_header(const std::filesystem::path& file,
                                      const ByteSource& source) noexcept
    {
        return Detail::at_boundary(K_HEADER_READER, [&]()
        {
            return read_header_checked(file, source);
        }, file);
    }

    Result<std::vector<std::uint8_t>> read_idx_payload(const std::filesystem::path& file,
                                                       const IDXHeader& header) noexcept
    {
        return read_idx_payload(file, header, Detail::FileSystemSource());
    }

    Result<std::vector<std::uint8_t>> read_idx_payload(const std::filesystem::path& file,
                                                       const IDXHeader& header,
                                                       const ByteSource& source) noexcept
    {
        return Detail::at_boundary(K_PAYLOAD_READER, [&]()
        {
            return read_payload_checked(file, header, source);
        }, file);
    }
} // namespace SushiAI::Vision::Datasets
