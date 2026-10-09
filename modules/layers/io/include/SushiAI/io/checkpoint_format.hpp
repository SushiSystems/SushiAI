/**************************************************************************/
/* checkpoint_format.hpp                                                  */
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
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>

namespace SushiAI
{
    namespace IO
    {
        /**
         * @brief Holds the first four bytes of every checkpoint, the ASCII "SACP", as the
         *        little-endian word a reader decodes them to.
         */
        inline constexpr std::uint32_t K_CHECKPOINT_MAGIC = 0x50434153u;

        /**
         * @brief Holds the format version this build reads and writes.
         *
         * A reader refuses a file that carries any other version.
         */
        inline constexpr std::uint32_t K_CHECKPOINT_VERSION = 1u;

        /** @brief Holds the largest tensor count a file may declare. */
        inline constexpr std::uint32_t K_MAX_TENSOR_COUNT = 4096u;

        /** @brief Holds the longest parameter name a file may declare, in bytes. */
        inline constexpr std::uint32_t K_MAX_NAME_LENGTH = 256u;

        /**
         * @struct TensorRecord
         * @brief Holds one tensor's identity in a checkpoint: its name, type and extents.
         */
        struct TensorRecord
        {
            /** @brief Holds the dotted parameter name, such as "0.weight". */
            std::string name;

            /** @brief Holds the element type. */
            DType dtype = DType::Float32;

            /** @brief Holds the extents. */
            Shape shape{};

            /**
             * @brief Returns the byte size of this record's payload.
             * @return The element count times the element size.
             */
            [[nodiscard]] std::size_t payload_bytes() const noexcept
            {
                return static_cast<std::size_t>(shape.num_elements()) * dtype_size(dtype);
            }
        };

        /**
         * @class CheckpointWriter
         * @brief Serializes checkpoint headers and tensor records to an output stream.
         */
        class CheckpointWriter final
        {
            public:
                /**
                 * @brief Writes the file header and returns a writer for the records.
                 * @param out Where the bytes go; must outlive the writer.
                 * @param count How many records will follow.
                 * @return The writer, or InvalidDescription when @p count exceeds
                 *         @ref K_MAX_TENSOR_COUNT, or InvalidFile when the header
                 *         cannot be written.
                 */
                [[nodiscard]] static Result<CheckpointWriter> create(std::ostream& out,
                                                                     std::uint32_t count);

                /**
                 * @brief Writes one record's header and its payload.
                 * @param record What the tensor is called, and its type and extents.
                 * @param payload @c record.payload_bytes() readable bytes.
                 * @return Success, InvalidDescription for a name longer than
                 *         @ref K_MAX_NAME_LENGTH, or InvalidFile when the write fails.
                 */
                [[nodiscard]] Status write(const TensorRecord& record, const void* payload);

                /**
                 * @brief Checks that exactly the declared number of records were written.
                 * @pre write() has succeeded once for each declared record.
                 */
                void finish() const;

                /** @brief Returns how many records this file declared. */
                [[nodiscard]] std::uint32_t declared_count() const noexcept
                {
                    return declared_;
                }

                /** @brief Returns how many records have been written so far. */
                [[nodiscard]] std::uint32_t written_count() const noexcept
                {
                    return written_;
                }

            private:
                /**
                 * @brief Stores the stream and the declared count; writes nothing.
                 */
                CheckpointWriter(std::ostream& out, std::uint32_t count) noexcept;

                /** @brief Holds the stream the header and records go to. */
                std::ostream* out_ = nullptr;

                /** @brief Holds the record count the header declared. */
                std::uint32_t declared_ = 0;

                /** @brief Holds the number of records written so far. */
                std::uint32_t written_ = 0;
        };

        /**
         * @class CheckpointReader
         * @brief Deserializes checkpoint headers and records from an input stream.
         */
        class CheckpointReader final
        {
            public:
                /**
                 * @brief Reads and validates the file header.
                 * @param in Where the bytes come from; must outlive the reader.
                 * @return The reader, or InvalidFile when the magic, the version or
                 *         the count is wrong, or the header is truncated.
                 */
                [[nodiscard]] static Result<CheckpointReader> open(std::istream& in);

                /** @brief Returns how many records the file declares. */
                [[nodiscard]] std::uint32_t count() const noexcept
                {
                    return count_;
                }

                /** @brief Returns how many records have been read so far. */
                [[nodiscard]] std::uint32_t read_count() const noexcept
                {
                    return read_;
                }

                /**
                 * @brief Reads the next record's header, leaving its payload unread.
                 * @return The record, or InvalidFile when a bound is exceeded, a
                 *         declared extent is not positive, the dtype is unknown, or
                 *         the header is truncated.
                 * @pre read_count() is less than count().
                 */
                [[nodiscard]] Result<TensorRecord> next();

                /**
                 * @brief Reads the payload of the record @ref next just returned.
                 * @param record The record @ref next returned.
                 * @param destination @c record.payload_bytes() writable bytes.
                 * @return Success, or InvalidFile when the stream ends early.
                 * @pre @p destination is not null.
                 */
                [[nodiscard]] Status read_payload(const TensorRecord& record, void* destination);

            private:
                /**
                 * @brief Stores the stream and the declared count; reads nothing.
                 */
                CheckpointReader(std::istream& in, std::uint32_t count) noexcept;

                /** @brief Holds the stream the header and records come from. */
                std::istream* in_ = nullptr;

                /** @brief Holds the record count the header declared. */
                std::uint32_t count_ = 0;

                /** @brief Holds the number of records read so far. */
                std::uint32_t read_ = 0;
        };

    } // namespace IO
} // namespace SushiAI
