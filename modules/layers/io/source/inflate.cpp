/**************************************************************************/
/* inflate.cpp                                                            */
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
 * @file inflate.cpp
 * @brief Defines the decoder of a raw DEFLATE stream, RFC 1951.
 * @author Mustafa Garip
 */

#include <SushiAI/io/inflate.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace IO
    {
        namespace
        {
            /** @brief Holds the longest code a Huffman table of DEFLATE holds, in bits. */
            constexpr unsigned K_MAX_BITS = 15;

            /** @brief Holds the symbols of the fixed literal and length code. */
            constexpr std::size_t K_FIXED_LITERAL_SYMBOLS = 288;

            /** @brief Holds the symbols of the fixed distance code, two of them unused. */
            constexpr std::size_t K_FIXED_DISTANCE_SYMBOLS = 32;

            /** @brief Holds the most literal and length codes a dynamic block states. */
            constexpr std::size_t K_MAX_LITERAL_CODES = 286;

            /** @brief Holds the most distance codes a dynamic block states. */
            constexpr std::size_t K_MAX_DISTANCE_CODES = 30;

            /** @brief Holds the symbols of the code that spells the code lengths. */
            constexpr std::size_t K_CODE_LENGTH_SYMBOLS = 19;

            /** @brief Holds the literal and length symbol that ends a block. */
            constexpr int K_END_OF_BLOCK = 256;

            /** @brief Holds the first literal and length symbol that names a match length. */
            constexpr int K_FIRST_LENGTH_SYMBOL = 257;

            /** @brief Holds the number of match lengths a symbol can name. */
            constexpr int K_LENGTH_SYMBOLS = 29;

            /** @brief Holds the stored-block length field's complement mask. */
            constexpr std::uint32_t K_STORED_MASK = 0xFFFFu;

            /** @brief Holds the reason given when the input ends before the stream does. */
            constexpr const char* K_ENDS = "the stream ends inside a block";

            /** @brief Holds the reason given when the output is too small for the stream. */
            constexpr const char* K_OVERRUN = "the block writes past the end of the output";

            /** @brief Holds the smallest match length of each length symbol, from 257. */
            constexpr std::uint16_t K_LENGTH_BASE[K_LENGTH_SYMBOLS] = {
                3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
                31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};

            /** @brief Holds the extra bits that follow each length symbol. */
            constexpr std::uint8_t K_LENGTH_EXTRA[K_LENGTH_SYMBOLS] = {
                0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};

            /** @brief Holds the smallest distance of each distance symbol. */
            constexpr std::uint16_t K_DISTANCE_BASE[K_MAX_DISTANCE_CODES] = {
                1,    2,    3,    4,    5,    7,    9,    13,   17,   25,
                33,   49,   65,   97,   129,  193,  257,  385,  513,  769,
                1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};

            /** @brief Holds the extra bits that follow each distance symbol. */
            constexpr std::uint8_t K_DISTANCE_EXTRA[K_MAX_DISTANCE_CODES] = {
                0, 0, 0, 0, 1, 1, 2, 2,  3,  3,  4,  4,  5,  5,  6,
                6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

            /** @brief Holds the order a dynamic block states its code length code in. */
            constexpr std::uint8_t K_CODE_LENGTH_ORDER[K_CODE_LENGTH_SYMBOLS] = {
                16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};

            /** @brief Holds a canonical Huffman code: its codes per length and its symbols in order. */
            struct Huffman
            {
                std::uint16_t count[K_MAX_BITS + 1]; /**< The codes of each length. */
                std::uint16_t symbol[K_FIXED_LITERAL_SYMBOLS]; /**< The symbols by code order. */
            };

            /**
             * @brief Builds the canonical code of @p count symbols with the lengths @p lengths.
             * @param complete_only Refuses a code that leaves some bit patterns unused.
             * @return The reason the lengths do not make a code, or null when they do. A code of
             *         no symbol is accepted, and so is an incomplete one that holds a single
             *         symbol of one bit, when @p complete_only is false.
             */
            const char* build(Huffman& code, const std::uint8_t* lengths, std::size_t count,
                              bool complete_only) noexcept
            {
                std::memset(code.count, 0, sizeof(code.count));
                for (std::size_t i = 0; i < count; ++i)
                    ++code.count[lengths[i]];
                if (code.count[0] == count)
                    return nullptr;

                int left = 1;
                for (unsigned length = 1; length <= K_MAX_BITS; ++length)
                {
                    left <<= 1;
                    left -= code.count[length];
                    if (left < 0)
                        return "a Huffman code has more codes than its lengths allow";
                }
                if (left > 0 && (complete_only || code.count[1] != count - code.count[0]))
                    return "a Huffman code leaves bit patterns unused";

                std::uint16_t offsets[K_MAX_BITS + 1];
                offsets[1] = 0;
                for (unsigned length = 1; length < K_MAX_BITS; ++length)
                    offsets[length + 1] = static_cast<std::uint16_t>(offsets[length] +
                                                                     code.count[length]);
                for (std::size_t i = 0; i < count; ++i)
                {
                    if (lengths[i] != 0)
                        code.symbol[offsets[lengths[i]]++] = static_cast<std::uint16_t>(i);
                }
                return nullptr;
            }

            /** @brief Reads a stream of bits, least significant bit of each byte first. */
            class BitReader final
            {
                public:
                    /** @brief Constructs a reader at the first bit of @p bytes. */
                    explicit BitReader(SushiRuntime::span<const std::byte> bytes) noexcept
                        : bytes_(bytes)
                    {

                    }

                    /**
                     * @brief Returns the next @p count bits, the first read in the lowest bit.
                     * @param count At most 16.
                     * @return The bits; 0 and a raised ran_out() when the input ends first.
                     */
                    std::uint32_t take(unsigned count) noexcept
                    {
                        while (held_ < count)
                        {
                            if (next_ >= bytes_.size())
                            {
                                ran_out_ = true;
                                return 0;
                            }
                            buffer_ |= static_cast<std::uint32_t>(
                                           static_cast<unsigned char>(bytes_[next_++]))
                                       << held_;
                            held_ += 8;
                        }
                        const std::uint32_t value = buffer_ & ((1u << count) - 1u);
                        buffer_ >>= count;
                        held_ -= count;
                        return value;
                    }

                    /** @brief Drops the bits left of the current byte. */
                    void align() noexcept
                    {
                        buffer_ = 0;
                        held_ = 0;
                    }

                    /**
                     * @brief Copies the next @p count whole bytes to @p destination.
                     * @pre The reader is aligned.
                     * @return False, and a raised ran_out(), when fewer bytes remain.
                     */
                    bool copy(std::byte* destination, std::size_t count) noexcept
                    {
                        if (count > bytes_.size() - next_)
                        {
                            ran_out_ = true;
                            return false;
                        }
                        if (count > 0)
                            std::memcpy(destination, bytes_.data() + next_, count);
                        next_ += count;
                        return true;
                    }

                    /** @brief Returns whether a read went past the end of the input. */
                    bool ran_out() const noexcept
                    {
                        return ran_out_;
                    }

                    /** @brief Returns the bits consumed so far. */
                    std::uint64_t consumed_bits() const noexcept
                    {
                        return static_cast<std::uint64_t>(next_) * 8u - held_;
                    }

                private:
                    /** @brief Holds the input. */
                    SushiRuntime::span<const std::byte> bytes_;

                    /** @brief Holds the index of the next input byte to load. */
                    std::size_t next_ = 0;

                    /** @brief Holds the loaded bits that are not yet consumed. */
                    std::uint32_t buffer_ = 0;

                    /** @brief Holds how many bits of buffer_ are not yet consumed. */
                    unsigned held_ = 0;

                    /** @brief Records that a read went past the end of the input. */
                    bool ran_out_ = false;
            };

            /**
             * @brief Decodes the next symbol of @p code from @p reader.
             * @return The symbol, or -1 when the bits spell no code.
             */
            int decode(BitReader& reader, const Huffman& code) noexcept
            {
                int bits = 0;
                int first = 0;
                int index = 0;
                for (unsigned length = 1; length <= K_MAX_BITS; ++length)
                {
                    bits |= static_cast<int>(reader.take(1));
                    const int count = code.count[length];
                    if (bits - count < first)
                        return code.symbol[index + (bits - first)];
                    index += count;
                    first += count;
                    first <<= 1;
                    bits <<= 1;
                }
                return -1;
            }

            /** @brief Decodes the blocks of one stream into one output. */
            class Inflater final
            {
                public:
                    /** @brief Constructs a decoder of @p compressed that writes @p output. */
                    Inflater(SushiRuntime::span<const std::byte> compressed,
                             SushiRuntime::span<std::byte> output) noexcept
                        : reader_(compressed), output_(output)
                    {

                    }

                    /** @brief Decodes every block up to the last; returns why it failed, or null. */
                    const char* run() noexcept
                    {
                        bool last = false;
                        while (!last)
                        {
                            last = reader_.take(1) != 0;
                            const std::uint32_t type = reader_.take(2);
                            if (reader_.ran_out())
                                return K_ENDS;
                            const char* reason = nullptr;
                            switch (type)
                            {
                                case 0: reason = stored_block(); break;
                                case 1: reason = fixed_block(); break;
                                case 2: reason = dynamic_block(); break;
                                default: reason = "a block has the reserved type 3"; break;
                            }
                            if (reason != nullptr)
                                return reason;
                        }
                        return nullptr;
                    }

                    /** @brief Returns the bytes written to the output. */
                    std::size_t written() const noexcept
                    {
                        return written_;
                    }

                    /** @brief Returns the bits of the stream consumed so far. */
                    std::uint64_t bit_offset() const noexcept
                    {
                        return reader_.consumed_bits();
                    }

                private:
                    /** @brief Copies a stored block to the output. */
                    const char* stored_block() noexcept
                    {
                        reader_.align();
                        const std::uint32_t length = reader_.take(16);
                        const std::uint32_t complement = reader_.take(16);
                        if (reader_.ran_out())
                            return K_ENDS;
                        if ((length ^ K_STORED_MASK) != complement)
                            return "a stored block states a length and a check that disagree";
                        if (length > output_.size() - written_)
                            return K_OVERRUN;
                        if (!reader_.copy(output_.data() + written_, length))
                            return K_ENDS;
                        written_ += length;
                        return nullptr;
                    }

                    /** @brief Decodes a block that uses the fixed code. */
                    const char* fixed_block() noexcept
                    {
                        std::uint8_t lengths[K_FIXED_LITERAL_SYMBOLS];
                        for (std::size_t i = 0; i < K_FIXED_LITERAL_SYMBOLS; ++i)
                            lengths[i] = i < 144 ? 8 : i < 256 ? 9 : i < 280 ? 7 : 8;
                        Huffman literal;
                        const char* reason = build(literal, lengths, K_FIXED_LITERAL_SYMBOLS, true);
                        if (reason != nullptr)
                            return reason;

                        std::uint8_t distance_lengths[K_FIXED_DISTANCE_SYMBOLS];
                        std::memset(distance_lengths, 5, sizeof(distance_lengths));
                        Huffman distance;
                        reason = build(distance, distance_lengths, K_FIXED_DISTANCE_SYMBOLS, true);
                        if (reason != nullptr)
                            return reason;
                        return codes_block(literal, distance);
                    }

                    /** @brief Reads the codes a dynamic block states, then decodes the block. */
                    const char* dynamic_block() noexcept
                    {
                        const std::size_t literal_count = reader_.take(5) + 257u;
                        const std::size_t distance_count = reader_.take(5) + 1u;
                        const std::size_t spelled = reader_.take(4) + 4u;
                        if (reader_.ran_out())
                            return K_ENDS;
                        if (literal_count > K_MAX_LITERAL_CODES ||
                            distance_count > K_MAX_DISTANCE_CODES)
                            return "a block states more length or distance codes than exist";

                        std::uint8_t spelling[K_CODE_LENGTH_SYMBOLS] = {};
                        for (std::size_t i = 0; i < spelled; ++i)
                            spelling[K_CODE_LENGTH_ORDER[i]] =
                                static_cast<std::uint8_t>(reader_.take(3));
                        if (reader_.ran_out())
                            return K_ENDS;
                        Huffman length_code;
                        const char* reason =
                            build(length_code, spelling, K_CODE_LENGTH_SYMBOLS, true);
                        if (reason != nullptr)
                            return reason;

                        std::uint8_t lengths[K_MAX_LITERAL_CODES + K_MAX_DISTANCE_CODES] = {};
                        const std::size_t total = literal_count + distance_count;
                        std::size_t index = 0;
                        while (index < total)
                        {
                            const int symbol = decode(reader_, length_code);
                            if (reader_.ran_out())
                                return K_ENDS;
                            if (symbol < 0)
                                return "the bits spell no code length";
                            if (symbol < 16)
                            {
                                lengths[index++] = static_cast<std::uint8_t>(symbol);
                                continue;
                            }
                            std::uint8_t value = 0;
                            std::size_t repeat = 0;
                            if (symbol == 16)
                            {
                                if (index == 0)
                                    return "a code length repeats a length that does not exist";
                                value = lengths[index - 1];
                                repeat = 3u + reader_.take(2);
                            }
                            else if (symbol == 17)
                            {
                                repeat = 3u + reader_.take(3);
                            }
                            else
                            {
                                repeat = 11u + reader_.take(7);
                            }
                            if (reader_.ran_out())
                                return K_ENDS;
                            if (repeat > total - index)
                                return "a repeat runs past the code lengths of the block";
                            for (; repeat > 0; --repeat)
                                lengths[index++] = value;
                        }
                        if (lengths[K_END_OF_BLOCK] == 0)
                            return "a dynamic block has no code for the end of the block";

                        Huffman literal;
                        reason = build(literal, lengths, literal_count, false);
                        if (reason != nullptr)
                            return reason;
                        Huffman distance;
                        reason = build(distance, lengths + literal_count, distance_count, false);
                        if (reason != nullptr)
                            return reason;
                        return codes_block(literal, distance);
                    }

                    /** @brief Decodes literals and matches to the end of the block. */
                    const char* codes_block(const Huffman& literal,
                                            const Huffman& distance) noexcept
                    {
                        std::byte* const out = output_.data();
                        for (;;)
                        {
                            int symbol = decode(reader_, literal);
                            if (reader_.ran_out())
                                return K_ENDS;
                            if (symbol < 0)
                                return "the bits spell no literal or length code";
                            if (symbol < K_END_OF_BLOCK)
                            {
                                if (written_ >= output_.size())
                                    return K_OVERRUN;
                                out[written_++] = static_cast<std::byte>(symbol);
                                continue;
                            }
                            if (symbol == K_END_OF_BLOCK)
                                return nullptr;

                            symbol -= K_FIRST_LENGTH_SYMBOL;
                            if (symbol >= K_LENGTH_SYMBOLS)
                                return "a length symbol is beyond the table";
                            const std::size_t length =
                                K_LENGTH_BASE[symbol] + reader_.take(K_LENGTH_EXTRA[symbol]);

                            const int distance_symbol = decode(reader_, distance);
                            if (reader_.ran_out())
                                return K_ENDS;
                            if (distance_symbol < 0 ||
                                distance_symbol >= static_cast<int>(K_MAX_DISTANCE_CODES))
                                return "the bits spell no distance code";
                            const std::size_t back =
                                K_DISTANCE_BASE[distance_symbol] +
                                reader_.take(K_DISTANCE_EXTRA[distance_symbol]);
                            if (reader_.ran_out())
                                return K_ENDS;
                            if (back > written_)
                                return "a match reaches back before the start of the output";
                            if (length > output_.size() - written_)
                                return K_OVERRUN;
                            for (std::size_t i = 0; i < length; ++i, ++written_)
                                out[written_] = out[written_ - back];
                        }
                    }

                    /** @brief Holds the bits of the stream. */
                    BitReader reader_;

                    /** @brief Holds the place the decoded bytes go. */
                    SushiRuntime::span<std::byte> output_;

                    /** @brief Holds the bytes written so far. */
                    std::size_t written_ = 0;
            };
        } // namespace

        Result<std::size_t> inflate(SushiRuntime::span<const std::byte> compressed,
                                    SushiRuntime::span<std::byte> output,
                                    std::string_view origin) noexcept
        {
            Inflater inflater(compressed, output);
            const char* const reason = inflater.run();
            SA_FAIL_IF(reason != nullptr, StatusCode::InvalidFile,
                       "'{}': the deflate stream is corrupt at bit {}: {}", origin,
                       inflater.bit_offset(), reason);
            return inflater.written();
        }
    } // namespace IO
} // namespace SushiAI
