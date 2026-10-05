/****************************************************************/
/* checkpoint_format.hpp                                        */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file checkpoint_format.hpp
 * @brief Defines the binary serialization format and streaming reader and writer.
 * @author Mustafa Garip
 */

#pragma once

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
         * @brief Serializes checkpoint headers and tensor records to an output stream.
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
         * @brief Deserializes checkpoint headers and records from an input stream.
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
