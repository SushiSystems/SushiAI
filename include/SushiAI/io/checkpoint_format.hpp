/**************************************************************************/
/* checkpoint_format.hpp                                                  */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                https://github.com/SushiSystems/SushiAI                 */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#pragma once

/**
 * @file checkpoint_format.hpp
 * @brief The checkpoint byte stream, and nothing else.
 *
 * @section layout The format
 *
 * @code
 * magic     u32       0x50434153, which is the bytes "SACP"
 * version   u32       1
 * count     u32       tensor count                  (<= MAX_TENSOR_COUNT)
 * --- count times ---
 * name_len  u32                                     (<= MAX_NAME_LENGTH)
 * name      u8 * name_len
 * dtype     u32       SushiAI::DType
 * rank      u32                                     (<= SushiAI::MAX_RANK)
 * dims      i64 * rank                              (each > 0)
 * payload   u8 * (product(dims) * dtype_size(dtype))
 * @endcode
 *
 * @section lane Why this half has no tensors in it
 *
 * Host lane: no SushiBLAS engine, no SYCL, no @c Tensor. It reads and writes
 * @ref TensorRecord headers and opaque payload bytes, which is what lets the
 * whole format be exercised against a @c std::stringstream with no device
 * present - so every malformed-file case is a @c Unit_ test rather than an
 * @c Integration_ one. @c io/checkpoint.hpp is the other half: it walks a
 * module's parameters and binds records to the pool's tensors.
 *
 * @section allocation Why the reader allocates nothing
 *
 * A reader that sized a buffer from a file-declared element count would let a
 * corrupt file ask for 16 GB, and it would ask for it before anything had
 * checked the count against a destination. There is no intermediate buffer
 * here at all: @ref CheckpointReader::next reads a header, the caller checks
 * it against the destination it already owns, and
 * @ref CheckpointReader::read_payload copies into that destination. The
 * file-driven allocation is absent by construction rather than guarded
 * against.
 *
 * @section endianness Little-endian, stated rather than assumed
 *
 * Every scalar is encoded and decoded a byte at a time, so the format does not
 * inherit the host's byte order by accident and a big-endian host reads a file
 * a little-endian host wrote. The cost is a handful of shifts per header
 * field, against a payload that is orders of magnitude larger.
 *
 * @section bounds Why every count is bounded
 *
 * @c name_len, @c rank and @c count each bound a loop or a read, and each is
 * read from the file. An unbounded one turns a corrupt byte into an
 * unbounded read; the bounds are stated here so a violation is a diagnostic
 * naming the field rather than a hang or an allocation failure.
 */

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>

namespace SushiAI
{
    namespace IO
    {
        /**
         * @brief The first four bytes of every checkpoint: the ASCII "SACP".
         *
         * Spelled as the little-endian word rather than a character array so
         * it compares against what @c get_u32 decodes without a second
         * conversion at each use.
         */
        inline constexpr std::uint32_t CHECKPOINT_MAGIC = 0x50434153u;

        /**
         * @brief The format version this build reads and writes.
         *
         * A file carrying any other version is an error rather than something
         * read with today's field order. That is the whole point of the field:
         * the format's next change must be loud, not a misread.
         */
        inline constexpr std::uint32_t CHECKPOINT_VERSION = 1u;

        /** @brief The largest tensor count a file may declare. */
        inline constexpr std::uint32_t MAX_TENSOR_COUNT = 4096u;

        /** @brief The longest parameter name a file may declare, in bytes. */
        inline constexpr std::uint32_t MAX_NAME_LENGTH = 256u;

        /**
         * @struct TensorRecord
         * @brief One tensor's identity in a checkpoint: its name, type and extents.
         *
         * The @ref shape is recorded rather than an element count, and that is
         * the difference between a checkpoint that catches a transposed weight
         * and one that loads [256, 6] happily into [6, 256].
         */
        struct TensorRecord
        {
            /** @brief The dotted parameter name the walk produced, e.g. "0.weight". */
            std::string name;

            /** @brief The element type. */
            DType dtype = DType::FLOAT32;

            /** @brief The extents. */
            Shape shape{};

            /**
             * @brief How many bytes this record's payload occupies.
             * @return The element count times the element size.
             */
            std::size_t payload_bytes() const noexcept
            {
                return static_cast<std::size_t>(shape.num_elements()) * dtype_size(dtype);
            }
        };

        /**
         * @class CheckpointWriter
         * @brief Writes the file header, then exactly @c count records.
         *
         * The count is declared up front rather than patched in at the end,
         * because a writer that seeks back to fix a header cannot write to a
         * non-seekable stream - and because a mismatch between the declared
         * and the actual count is then a condition @ref finish can name.
         */
        class CheckpointWriter
        {
            public:
                /**
                 * @brief Writes the file header immediately.
                 * @param out Where the bytes go; must outlive the writer.
                 * @param count How many records will follow.
                 * @throws Error If @p count exceeds @ref MAX_TENSOR_COUNT, or the write fails.
                 */
                CheckpointWriter(std::ostream& out, std::uint32_t count);

                /**
                 * @brief Writes one record's header and its payload.
                 * @param record What the tensor is called, and its type and extents.
                 * @param payload @c record.payload_bytes() readable bytes.
                 * @throws Error If @p record breaks a bound, more records are
                 *         written than were declared, or the write fails.
                 */
                void write(const TensorRecord& record, const void* payload);

                /**
                 * @brief Checks that exactly the declared number of records were written.
                 * @throws Error If fewer were written, naming both counts.
                 */
                void finish() const;

                /** @brief How many records this file declared. */
                std::uint32_t declared_count() const noexcept { return declared_; }

                /** @brief How many records have been written so far. */
                std::uint32_t written_count() const noexcept { return written_; }

            private:
                std::ostream* out_ = nullptr;
                std::uint32_t declared_ = 0;
                std::uint32_t written_ = 0;
        };

        /**
         * @class CheckpointReader
         * @brief Reads the file header, then hands out records one at a time.
         *
         * @ref next and @ref read_payload are deliberately two calls: between
         * them is the only place a caller can compare a record against the
         * destination it owns, which is what lets the payload be read straight
         * into that destination instead of into a buffer sized from the file.
         */
        class CheckpointReader
        {
            public:
                /**
                 * @brief Reads and validates the file header.
                 * @param in Where the bytes come from; must outlive the reader.
                 * @throws Error If the magic, the version or the count is wrong,
                 *         or the header is truncated.
                 */
                explicit CheckpointReader(std::istream& in);

                /** @brief How many records the file declares. */
                std::uint32_t count() const noexcept { return count_; }

                /** @brief How many records have been read so far. */
                std::uint32_t read_count() const noexcept { return read_; }

                /**
                 * @brief Reads the next record's header, leaving its payload unread.
                 * @return The record.
                 * @throws Error If no records remain, a bound is exceeded, a
                 *         declared extent is not positive, the dtype is not a
                 *         @c DType, or the header is truncated.
                 */
                TensorRecord next();

                /**
                 * @brief Reads the payload of the record @ref next just returned.
                 * @param record The record @ref next returned.
                 * @param destination @c record.payload_bytes() writable bytes.
                 * @throws Error If @p destination is null, or the stream ends early.
                 */
                void read_payload(const TensorRecord& record, void* destination);

            private:
                std::istream* in_ = nullptr;
                std::uint32_t count_ = 0;
                std::uint32_t read_ = 0;
        };

    } // namespace IO
} // namespace SushiAI
