/**************************************************************************/
/* npy_reader.cpp                                                         */
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
 * @file npy_reader.cpp
 * @brief Defines the reader of a NumPy .npy file and the widening of its array to float32.
 * @author Mustafa Garip
 */

#include <SushiAI/io/npy_reader.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/shape/shape.hpp>
#include "float_widening.hpp"

namespace SushiAI
{
    namespace IO
    {
        namespace
        {
            /** @brief Pairs a type with the descriptor the header spells it as. */
            struct TypeRow
            {
                NpyType type;           /**< The element type. */
                std::string_view name;  /**< Its descriptor. */
                std::size_t bytes;      /**< The bytes of one element. */
            };

            /** @brief Lists every element type the reader reads. */
            constexpr TypeRow K_TYPES[] = {
                {NpyType::Float16, "<f2", 2}, {NpyType::Float32, "<f4", 4},
                {NpyType::Float64, "<f8", 8}, {NpyType::Int8, "|i1", 1},
                {NpyType::UInt8, "|u1", 1},   {NpyType::Int16, "<i2", 2},
                {NpyType::UInt16, "<u2", 2},  {NpyType::Int32, "<i4", 4},
                {NpyType::UInt32, "<u4", 4},  {NpyType::Int64, "<i8", 8},
                {NpyType::UInt64, "<u8", 8},
            };

            /** @brief Holds the bytes every .npy file opens with. */
            constexpr std::string_view K_MAGIC = "\x93NUMPY";

            /** @brief Holds the bytes before a version 1 header: magic, version and length. */
            constexpr std::size_t K_PREAMBLE_V1 = 10;

            /** @brief Holds the bytes before a version 2 or 3 header. */
            constexpr std::size_t K_PREAMBLE_V2 = 12;

            /** @brief Holds the largest extent the reader accepts. */
            constexpr std::int64_t K_MAX_EXTENT = std::int64_t{1} << 40;

            /** @brief Holds the most elements an array may hold, so no byte count overflows. */
            constexpr std::uint64_t K_MAX_ELEMENTS = std::uint64_t{1} << 40;

            /** @brief Reads the little-endian integer of @p width bytes at @p at. */
            std::uint64_t read_le(SushiRuntime::span<const std::byte> bytes, std::size_t at,
                                  std::size_t width) noexcept
            {
                std::uint64_t value = 0;
                for (std::size_t i = 0; i < width; ++i)
                    value |= static_cast<std::uint64_t>(bytes[at + i]) << (8u * i);
                return value;
            }

            /** @brief Holds the three entries of the header dictionary. */
            struct NpyHeader
            {
                std::string descr;                  /**< The element type descriptor. */
                bool fortran_order = false;         /**< Whether the data is column-major. */
                std::vector<std::int64_t> extents;  /**< The extents of the array. */
            };

            /** @brief Reads the Python dictionary literal a .npy header holds. */
            class HeaderParser final
            {
                public:
                    /** @brief Constructs a parser of @p text; @p origin names the file. */
                    HeaderParser(std::string_view text, std::string_view origin) noexcept
                        : text_(text), origin_(origin)
                    {

                    }

                    /**
                     * @brief Reads the dictionary.
                     * @return InvalidFile for text that is not a dictionary of exactly "descr",
                     *         "fortran_order" and "shape", each once; Unsupported for a
                     *         structured descriptor.
                     */
                    Result<NpyHeader> parse()
                    {
                        NpyHeader header;
                        bool has_descr = false;
                        bool has_order = false;
                        bool has_shape = false;
                        SA_TRY(expect("{"));
                        for (;;)
                        {
                            if (accept("}"))
                                break;
                            SA_TRY_ASSIGN(const std::string key, read_string());
                            SA_TRY(expect(":"));
                            if (key == "descr")
                            {
                                SA_FAIL_IF(has_descr, StatusCode::InvalidFile,
                                           "'{}': the header states 'descr' twice", origin_);
                                SA_FAIL_IF(peek("["), StatusCode::Unsupported,
                                           "'{}': the array has a structured type, which is not "
                                           "read",
                                           origin_);
                                SA_TRY_ASSIGN(header.descr, read_string());
                                has_descr = true;
                            }
                            else if (key == "fortran_order")
                            {
                                SA_FAIL_IF(has_order, StatusCode::InvalidFile,
                                           "'{}': the header states 'fortran_order' twice",
                                           origin_);
                                SA_TRY_ASSIGN(header.fortran_order, read_bool());
                                has_order = true;
                            }
                            else if (key == "shape")
                            {
                                SA_FAIL_IF(has_shape, StatusCode::InvalidFile,
                                           "'{}': the header states 'shape' twice", origin_);
                                SA_TRY_ASSIGN(header.extents, read_tuple());
                                has_shape = true;
                            }
                            else
                            {
                                SA_FAIL_IF(true, StatusCode::InvalidFile,
                                           "'{}': the header holds the key '{}', and only "
                                           "'descr', 'fortran_order' and 'shape' are allowed",
                                           origin_, key);
                            }
                            if (accept(","))
                                continue;
                            SA_TRY(expect("}"));
                            break;
                        }
                        skip_space();
                        SA_FAIL_IF(at_ != text_.size(), StatusCode::InvalidFile,
                                   "'{}': the header holds text after its dictionary, at its "
                                   "byte {}",
                                   origin_, at_);
                        SA_FAIL_IF(!has_descr || !has_order || !has_shape, StatusCode::InvalidFile,
                                   "'{}': the header lacks one of 'descr', 'fortran_order' and "
                                   "'shape'",
                                   origin_);
                        return header;
                    }

                private:
                    /** @brief Moves past spaces, tabs and line ends. */
                    void skip_space() noexcept
                    {
                        while (at_ < text_.size() &&
                               (text_[at_] == ' ' || text_[at_] == '\n' || text_[at_] == '\t' ||
                                text_[at_] == '\r'))
                            ++at_;
                    }

                    /** @brief Returns whether the text continues with @p token after any spaces. */
                    bool peek(std::string_view token) noexcept
                    {
                        skip_space();
                        return text_.substr(at_, token.size()) == token;
                    }

                    /** @brief Moves past @p token when the text continues with it. */
                    bool accept(std::string_view token) noexcept
                    {
                        if (!peek(token))
                            return false;
                        at_ += token.size();
                        return true;
                    }

                    /** @brief Moves past @p token, or fails when the text lacks it. */
                    Status expect(std::string_view token)
                    {
                        SA_FAIL_IF(!accept(token), StatusCode::InvalidFile,
                                   "'{}': the header has no '{}' at its byte {}", origin_, token,
                                   at_);
                        return Status();
                    }

                    /** @brief Reads a quoted string with no escapes. */
                    Result<std::string> read_string()
                    {
                        skip_space();
                        SA_FAIL_IF(at_ >= text_.size() || (text_[at_] != '\'' && text_[at_] != '"'),
                                   StatusCode::InvalidFile,
                                   "'{}': the header has no string at its byte {}", origin_, at_);
                        const char quote = text_[at_];
                        const std::size_t end = text_.find(quote, at_ + 1);
                        SA_FAIL_IF(end == std::string_view::npos, StatusCode::InvalidFile,
                                   "'{}': the string at byte {} of the header never closes",
                                   origin_, at_);
                        const std::string value(text_.substr(at_ + 1, end - at_ - 1));
                        SA_FAIL_IF(value.find('\\') != std::string::npos, StatusCode::InvalidFile,
                                   "'{}': the string at byte {} of the header holds an escape",
                                   origin_, at_);
                        at_ = end + 1;
                        return value;
                    }

                    /** @brief Reads True or False. */
                    Result<bool> read_bool()
                    {
                        if (accept("True"))
                            return true;
                        SA_FAIL_IF(!accept("False"), StatusCode::InvalidFile,
                                   "'{}': the header has no True or False at its byte {}",
                                   origin_, at_);
                        return false;
                    }

                    /** @brief Reads a tuple of whole numbers, such as (3, 4), (5,) or (). */
                    Result<std::vector<std::int64_t>> read_tuple()
                    {
                        SA_TRY(expect("("));
                        std::vector<std::int64_t> values;
                        while (!accept(")"))
                        {
                            skip_space();
                            const std::size_t first = at_;
                            std::int64_t value = 0;
                            while (at_ < text_.size() && text_[at_] >= '0' && text_[at_] <= '9')
                            {
                                value = value * 10 + (text_[at_] - '0');
                                SA_FAIL_IF(value > K_MAX_EXTENT, StatusCode::Unsupported,
                                           "'{}': the shape holds an extent above {}", origin_,
                                           K_MAX_EXTENT);
                                ++at_;
                            }
                            SA_FAIL_IF(at_ == first, StatusCode::InvalidFile,
                                       "'{}': the shape holds no whole number at header byte {}",
                                       origin_, at_);
                            if (at_ < text_.size() && text_[at_] == 'L')
                                ++at_;
                            values.push_back(value);
                            if (!accept(","))
                            {
                                SA_TRY(expect(")"));
                                break;
                            }
                        }
                        return values;
                    }

                    /** @brief Holds the header's text. */
                    std::string_view text_;

                    /** @brief Names the file in a message. */
                    std::string_view origin_;

                    /** @brief Holds the index of the next character to read. */
                    std::size_t at_ = 0;
            };

            /**
             * @brief Returns the row of the descriptor @p descr.
             * @return The row; Unsupported naming @p descr for an object array, a big-endian type
             *         and any type the reader lacks.
             */
            Result<const TypeRow*> find_type(const std::string& descr, std::string_view origin)
            {
                SA_FAIL_IF(descr.size() == 2 && descr[1] == 'O', StatusCode::Unsupported,
                           "'{}': the array is an object array ('{}'), which holds Python objects "
                           "and is not read",
                           origin, descr);
                SA_FAIL_IF(!descr.empty() && descr[0] == '>', StatusCode::Unsupported,
                           "'{}': the array is big-endian ('{}'), and only little-endian arrays "
                           "are read",
                           origin, descr);
                for (const TypeRow& row : K_TYPES)
                {
                    if (row.name == descr)
                        return &row;
                }
                SA_FAIL_IF(true, StatusCode::Unsupported,
                           "'{}': the array has the type '{}', which the reader lacks", origin,
                           descr);
                return nullptr;
            }

            /** @brief Widens one element of the C type @p Element. */
            template<typename Element>
            struct NativeLoad
            {
                /** @brief Holds the bytes of one element. */
                static constexpr std::size_t K_WIDTH = sizeof(Element);

                /** @brief Returns the element at @p raw as a float. */
                static float load(const std::byte* raw) noexcept
                {
                    Element value;
                    std::memcpy(&value, raw, sizeof(Element));
                    return static_cast<float>(value);
                }
            };

            /** @brief Widens one binary16 element. */
            struct HalfLoad
            {
                /** @brief Holds the bytes of one element. */
                static constexpr std::size_t K_WIDTH = 2;

                /** @brief Returns the element at @p raw as a float. */
                static float load(const std::byte* raw) noexcept
                {
                    const std::uint16_t bits = static_cast<std::uint16_t>(
                        static_cast<unsigned>(raw[0]) | (static_cast<unsigned>(raw[1]) << 8u));
                    return Detail::widen_half(bits);
                }
            };

            /**
             * @brief Writes the elements of @p data to @p out in row-major order.
             * @param Load The element type's widening.
             * @param fortran Whether @p data is column-major.
             */
            template<typename Load>
            void widen(const std::byte* data, const Shape& shape, bool fortran,
                       float* out) noexcept
            {
                const std::int64_t count = shape.num_elements();
                const std::int32_t rank = shape.rank();
                if (!fortran || rank < 2)
                {
                    for (std::int64_t n = 0; n < count; ++n)
                        out[n] = Load::load(data + static_cast<std::size_t>(n) * Load::K_WIDTH);
                    return;
                }

                // Axis a of a column-major array steps by the product of the extents before it.
                std::int64_t stride[K_MAX_RANK] = {};
                std::int64_t index[K_MAX_RANK] = {};
                stride[0] = 1;
                for (std::int32_t axis = 1; axis < rank; ++axis)
                    stride[axis] = stride[axis - 1] * shape[axis - 1];
                for (std::int64_t n = 0; n < count; ++n)
                {
                    std::int64_t element = 0;
                    for (std::int32_t axis = 0; axis < rank; ++axis)
                        element += index[axis] * stride[axis];
                    out[n] = Load::load(data + static_cast<std::size_t>(element) * Load::K_WIDTH);
                    for (std::int32_t axis = rank; axis-- > 0;)
                    {
                        if (++index[axis] < shape[axis])
                            break;
                        index[axis] = 0;
                    }
                }
            }
        } // namespace

        std::size_t npy_type_size(NpyType type) noexcept
        {
            for (const TypeRow& row : K_TYPES)
            {
                if (row.type == type)
                    return row.bytes;
            }
            return 0;
        }

        std::string_view npy_type_name(NpyType type) noexcept
        {
            for (const TypeRow& row : K_TYPES)
            {
                if (row.type == type)
                    return row.name;
            }
            return "";
        }

        NpyReader::NpyReader(SushiRuntime::span<const std::byte> bytes, NpyArray array) noexcept
            : bytes_(bytes), array_(array)
        {

        }

        Result<NpyReader> NpyReader::read(SushiRuntime::span<const std::byte> bytes,
                                          std::string_view origin) noexcept
        {
            SA_FAIL_IF(bytes.size() < K_PREAMBLE_V1 ||
                           std::memcmp(bytes.data(), K_MAGIC.data(), K_MAGIC.size()) != 0,
                       StatusCode::InvalidFile, "'{}' is not a .npy file: it lacks the magic bytes",
                       origin);
            const std::uint64_t major = read_le(bytes, 6, 1);
            const std::uint64_t minor = read_le(bytes, 7, 1);
            SA_FAIL_IF((major != 1 && major != 2 && major != 3) || minor != 0,
                       StatusCode::Unsupported,
                       "'{}': the .npy format version {}.{} is not read; versions 1.0, 2.0 and "
                       "3.0 are",
                       origin, major, minor);
            const std::size_t preamble = major == 1 ? K_PREAMBLE_V1 : K_PREAMBLE_V2;
            SA_FAIL_IF(bytes.size() < preamble, StatusCode::InvalidFile,
                       "'{}': the file ends inside its preamble", origin);
            const std::uint64_t header_length = read_le(bytes, 8, major == 1 ? 2 : 4);
            SA_FAIL_IF(header_length > bytes.size() - preamble, StatusCode::InvalidFile,
                       "'{}': the header states {} bytes, and the file holds {} after its "
                       "preamble",
                       origin, header_length, bytes.size() - preamble);

            const std::string_view text(reinterpret_cast<const char*>(bytes.data()) + preamble,
                                        static_cast<std::size_t>(header_length));
            SA_TRY_ASSIGN(const NpyHeader header, HeaderParser(text, origin).parse());
            SA_TRY_ASSIGN(const TypeRow* const row, find_type(header.descr, origin));

            SA_FAIL_IF(header.extents.size() > static_cast<std::size_t>(K_MAX_RANK),
                       StatusCode::Unsupported, "'{}': the array has rank {}, above the limit {}",
                       origin, header.extents.size(), K_MAX_RANK);
            std::uint64_t elements = 1;
            for (const std::int64_t extent : header.extents)
            {
                SA_FAIL_IF(extent == 0, StatusCode::Unsupported,
                           "'{}': the array has an extent of 0, and an empty array is not read",
                           origin);
                SA_FAIL_IF(elements > K_MAX_ELEMENTS / static_cast<std::uint64_t>(extent),
                           StatusCode::Unsupported,
                           "'{}': the array holds more than {} elements", origin, K_MAX_ELEMENTS);
                elements *= static_cast<std::uint64_t>(extent);
            }
            SA_TRY_ASSIGN(const Shape shape,
                          Shape::make(SushiRuntime::span<const std::int64_t>(
                              header.extents.data(), header.extents.size())));

            const std::size_t data_offset = preamble + static_cast<std::size_t>(header_length);
            const std::uint64_t length = elements * row->bytes;
            SA_FAIL_IF(length > bytes.size() - data_offset, StatusCode::InvalidFile,
                       "'{}': the array needs {} bytes of data, and the file holds {}", origin,
                       length, bytes.size() - data_offset);

            NpyArray array;
            array.type = row->type;
            array.shape = shape;
            array.fortran_order = header.fortran_order;
            array.offset = data_offset;
            array.length = static_cast<std::size_t>(length);
            return NpyReader(bytes, array);
        }

        const NpyArray& NpyReader::array() const noexcept
        {
            return array_;
        }

        SushiRuntime::span<const std::byte> NpyReader::contents() const noexcept
        {
            return bytes_.subspan(array_.offset, array_.length);
        }

        std::vector<float> NpyReader::read_float32() const noexcept
        {
            std::vector<float> out(static_cast<std::size_t>(array_.shape.num_elements()));
            const std::byte* const data = bytes_.data() + array_.offset;
            const bool fortran = array_.fortran_order;
            const Shape& shape = array_.shape;
            float* const target = out.data();
            switch (array_.type)
            {
                case NpyType::Float16:
                    widen<HalfLoad>(data, shape, fortran, target);
                    break;
                case NpyType::Float32:
                    widen<NativeLoad<float>>(data, shape, fortran, target);
                    break;
                case NpyType::Float64:
                    widen<NativeLoad<double>>(data, shape, fortran, target);
                    break;
                case NpyType::Int8:
                    widen<NativeLoad<std::int8_t>>(data, shape, fortran, target);
                    break;
                case NpyType::UInt8:
                    widen<NativeLoad<std::uint8_t>>(data, shape, fortran, target);
                    break;
                case NpyType::Int16:
                    widen<NativeLoad<std::int16_t>>(data, shape, fortran, target);
                    break;
                case NpyType::UInt16:
                    widen<NativeLoad<std::uint16_t>>(data, shape, fortran, target);
                    break;
                case NpyType::Int32:
                    widen<NativeLoad<std::int32_t>>(data, shape, fortran, target);
                    break;
                case NpyType::UInt32:
                    widen<NativeLoad<std::uint32_t>>(data, shape, fortran, target);
                    break;
                case NpyType::Int64:
                    widen<NativeLoad<std::int64_t>>(data, shape, fortran, target);
                    break;
                case NpyType::UInt64:
                    widen<NativeLoad<std::uint64_t>>(data, shape, fortran, target);
                    break;
            }
            return out;
        }
    } // namespace IO
} // namespace SushiAI
