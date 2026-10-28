/**************************************************************************/
/* idx_file.hpp                                                           */
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
 * @file idx_file.hpp
 * @brief Declares the header of an IDX file and the readers of its header and payload.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include <SushiAI/failure/result.hpp>

/** @brief Holds the readers of the on-disk layouts vision datasets are kept in. */
namespace SushiAI::Vision::Datasets
{
    /** @brief Names the element type of an IDX file by the code its third byte holds. */
    enum class IDXElement : std::uint8_t
    {
        UInt8 = 0x08,   /**< Unsigned byte. */
        Int8 = 0x09,    /**< Signed byte. */
        Int16 = 0x0B,   /**< 16-bit integer, most significant byte first. */
        Int32 = 0x0C,   /**< 32-bit integer, most significant byte first. */
        Float32 = 0x0D, /**< 32-bit IEEE 754 number, most significant byte first. */
        Float64 = 0x0E, /**< 64-bit IEEE 754 number, most significant byte first. */
    };

    /** @brief Returns the byte count of one element, or 0 for a code the format lacks. */
    [[nodiscard]] constexpr std::size_t idx_element_bytes(IDXElement element) noexcept
    {
        switch (element)
        {
            case IDXElement::UInt8:   return 1;
            case IDXElement::Int8:    return 1;
            case IDXElement::Int16:   return 2;
            case IDXElement::Int32:   return 4;
            case IDXElement::Float32: return 4;
            case IDXElement::Float64: return 8;
        }
        return 0;
    }

    /** @brief Holds the validated header of an IDX file. */
    struct IDXHeader
    {
        /** @brief Names the element type. */
        IDXElement element = IDXElement::UInt8;

        /** @brief Holds one extent per dimension, the outermost first; at least one. */
        std::vector<std::uint32_t> extents;

        /** @brief Holds the byte count of the elements, which fill the file after the header. */
        std::uint64_t payload_bytes = 0;

        /** @brief Returns the byte count of the header: four bytes, and four per dimension. */
        [[nodiscard]] std::uint64_t header_bytes() const noexcept
        {
            return 4u + 4u * static_cast<std::uint64_t>(extents.size());
        }
    };

    /**
     * @brief Reads and validates the header of an IDX file.
     * @return The header; FileNotFound when the path names nothing; InvalidFile when it is
     *         not a readable file, the first two bytes are not zero, the element code is
     *         unknown, there is no dimension, or the file's size is not the header plus the
     *         elements the header declares. Every message opens with the path.
     */
    [[nodiscard]] Result<IDXHeader> read_idx_header(const std::filesystem::path& file) noexcept;

    /**
     * @brief Reads every element byte of an IDX file, in the order the file stores them.
     * @param header The value read_idx_header returned for this file.
     * @return header.payload_bytes bytes; InvalidFile when the file cannot be opened or
     *         ends early; OutOfMemory. Every message opens with the path.
     */
    [[nodiscard]] Result<std::vector<std::uint8_t>> read_idx_payload(
        const std::filesystem::path& file, const IDXHeader& header) noexcept;
} // namespace SushiAI::Vision::Datasets
