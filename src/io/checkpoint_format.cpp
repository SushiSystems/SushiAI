/**************************************************************************/
/* checkpoint_format.cpp                                                  */
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

#include <SushiAI/io/checkpoint_format.hpp>

#include <array>
#include <istream>
#include <ostream>
#include <string_view>
#include <vector>

namespace SushiAI
{
    namespace IO
    {
        namespace
        {
            /**
             * @brief Every DType a checkpoint may name.
             *
             * A file-declared dtype is an arbitrary 32-bit number until it has
             * been checked against this list, and casting an unchecked one to
             * @c DType would put a value into the enum that no switch in the
             * library handles. Listed rather than range-checked so that adding
             * a dtype is an edit here beside the edits to @c dtype_size and
             * @c dtype_name, not a bound that silently widens.
             */
            constexpr std::array<DType, 7> KNOWN_DTYPES = {
                DType::HALF, DType::FLOAT32, DType::FLOAT64,
                DType::COMPLEX32, DType::COMPLEX64,
                DType::INT32, DType::INT64,
            };

            /**
             * @brief Whether @p encoded names a DType this build knows.
             * @param encoded The raw value read from the file.
             * @return True when it matches an entry of @ref KNOWN_DTYPES.
             */
            bool is_known_dtype(std::uint32_t encoded) noexcept
            {
                for (DType dtype : KNOWN_DTYPES)
                    if (static_cast<std::uint32_t>(dtype) == encoded) return true;
                return false;
            }

            /** @brief Writes @p value as four little-endian bytes. */
            void put_u32(std::ostream& out, std::uint32_t value)
            {
                char bytes[4];
                for (int i = 0; i < 4; ++i)
                    bytes[i] = static_cast<char>((value >> (8 * i)) & 0xFFu);
                out.write(bytes, 4);
            }

            /** @brief Writes @p value as eight little-endian bytes, two's complement. */
            void put_i64(std::ostream& out, std::int64_t value)
            {
                const std::uint64_t bits = static_cast<std::uint64_t>(value);
                char bytes[8];
                for (int i = 0; i < 8; ++i)
                    bytes[i] = static_cast<char>((bits >> (8 * i)) & 0xFFu);
                out.write(bytes, 8);
            }

            /**
             * @brief Reads four little-endian bytes.
             * @param in Where the bytes come from.
             * @param what What is being read, for the truncation diagnostic.
             * @return The decoded value.
             * @throws Error If the stream ends first.
             */
            std::uint32_t get_u32(std::istream& in, std::string_view what)
            {
                char bytes[4];
                in.read(bytes, 4);
                SA_THROW_IF(!in, "the checkpoint ended while reading {}", what);

                std::uint32_t value = 0;
                for (int i = 0; i < 4; ++i)
                    value |= static_cast<std::uint32_t>(static_cast<unsigned char>(bytes[i])) << (8 * i);
                return value;
            }

            /**
             * @brief Reads eight little-endian bytes as a signed value.
             * @param in Where the bytes come from.
             * @param what What is being read, for the truncation diagnostic.
             * @return The decoded value.
             * @throws Error If the stream ends first.
             */
            std::int64_t get_i64(std::istream& in, std::string_view what)
            {
                char bytes[8];
                in.read(bytes, 8);
                SA_THROW_IF(!in, "the checkpoint ended while reading {}", what);

                std::uint64_t bits = 0;
                for (int i = 0; i < 8; ++i)
                    bits |= static_cast<std::uint64_t>(static_cast<unsigned char>(bytes[i])) << (8 * i);
                return static_cast<std::int64_t>(bits);
            }
        }

        // --- CheckpointWriter ------------------------------------------------

        CheckpointWriter::CheckpointWriter(std::ostream& out, std::uint32_t count)
            : out_(&out), declared_(count)
        {
            SA_THROW_IF(count > MAX_TENSOR_COUNT,
                        "a checkpoint may hold at most {} tensors; {} were declared",
                        MAX_TENSOR_COUNT, count);

            put_u32(out, CHECKPOINT_MAGIC);
            put_u32(out, CHECKPOINT_VERSION);
            put_u32(out, count);
            SA_THROW_IF(!out, "failed writing the checkpoint header");
        }

        void CheckpointWriter::write(const TensorRecord& record, const void* payload)
        {
            SA_THROW_IF(written_ >= declared_,
                        "this checkpoint declared {} records and a further one "
                        "('{}') was written", declared_, record.name);
            SA_THROW_IF(payload == nullptr,
                        "'{}' was written with no payload", record.name);
            SA_THROW_IF(record.name.size() > MAX_NAME_LENGTH,
                        "'{}...' is {} bytes long; a checkpoint name may be at most {}",
                        record.name.substr(0, 32), record.name.size(), MAX_NAME_LENGTH);
            SA_THROW_IF(record.shape.rank() > MAX_RANK,
                        "'{}' is rank {}; a checkpoint tensor may be at most rank {}",
                        record.name, record.shape.rank(), MAX_RANK);

            put_u32(*out_, static_cast<std::uint32_t>(record.name.size()));
            out_->write(record.name.data(), static_cast<std::streamsize>(record.name.size()));
            put_u32(*out_, static_cast<std::uint32_t>(record.dtype));
            put_u32(*out_, static_cast<std::uint32_t>(record.shape.rank()));
            for (int32_t i = 0; i < record.shape.rank(); ++i)
                put_i64(*out_, record.shape[i]);

            out_->write(static_cast<const char*>(payload),
                        static_cast<std::streamsize>(record.payload_bytes()));
            SA_THROW_IF(!*out_, "failed writing '{}' to the checkpoint", record.name);

            ++written_;
        }

        void CheckpointWriter::finish() const
        {
            SA_THROW_IF(written_ != declared_,
                        "this checkpoint declared {} records but {} were written",
                        declared_, written_);
        }

        // --- CheckpointReader ------------------------------------------------

        CheckpointReader::CheckpointReader(std::istream& in) : in_(&in)
        {
            const std::uint32_t magic = get_u32(in, "the magic number");
            SA_THROW_IF(magic != CHECKPOINT_MAGIC,
                        "this is not a SushiAI checkpoint: expected the magic bytes "
                        "'SACP' (0x{:08X}), got 0x{:08X}", CHECKPOINT_MAGIC, magic);

            const std::uint32_t version = get_u32(in, "the version");
            SA_THROW_IF(version != CHECKPOINT_VERSION,
                        "this checkpoint is version {}; this build reads version {}",
                        version, CHECKPOINT_VERSION);

            count_ = get_u32(in, "the tensor count");
            SA_THROW_IF(count_ > MAX_TENSOR_COUNT,
                        "this checkpoint declares {} tensors; at most {} are allowed",
                        count_, MAX_TENSOR_COUNT);
        }

        TensorRecord CheckpointReader::next()
        {
            SA_THROW_IF(read_ >= count_,
                        "this checkpoint declared {} records and a further one was requested",
                        count_);

            const std::uint32_t name_length = get_u32(*in_, "a record's name length");
            SA_THROW_IF(name_length > MAX_NAME_LENGTH,
                        "a checkpoint record declares a {}-byte name; at most {} are allowed",
                        name_length, MAX_NAME_LENGTH);

            TensorRecord record;
            record.name.resize(name_length);
            if (name_length != 0)
            {
                in_->read(record.name.data(), static_cast<std::streamsize>(name_length));
                SA_THROW_IF(!*in_, "the checkpoint ended while reading a record's name");
            }

            const std::uint32_t encoded_dtype = get_u32(*in_, "'" + record.name + "'s dtype");
            SA_THROW_IF(!is_known_dtype(encoded_dtype),
                        "'{}' declares dtype {}, which this build does not know",
                        record.name, encoded_dtype);
            record.dtype = static_cast<DType>(encoded_dtype);

            const std::uint32_t rank = get_u32(*in_, "'" + record.name + "'s rank");
            SA_THROW_IF(rank > static_cast<std::uint32_t>(MAX_RANK),
                        "'{}' declares rank {}; at most {} is allowed",
                        record.name, rank, MAX_RANK);

            // Collected before the Shape is built, so a non-positive extent is
            // reported against the record that carries it rather than through
            // Shape's own constructor, which knows no name.
            std::array<std::int64_t, static_cast<std::size_t>(MAX_RANK)> dims{};
            for (std::uint32_t i = 0; i < rank; ++i)
            {
                dims[i] = get_i64(*in_, "'" + record.name + "'s extents");
                SA_THROW_IF(dims[i] <= 0,
                            "'{}' declares extent {} for dimension {}; every extent must be positive",
                            record.name, dims[i], i);
            }
            record.shape = Shape(SushiRuntime::span<const int64_t>(dims.data(), rank));

            ++read_;
            return record;
        }

        void CheckpointReader::read_payload(const TensorRecord& record, void* destination)
        {
            SA_THROW_IF(destination == nullptr,
                        "'{}' was read with no destination", record.name);

            in_->read(static_cast<char*>(destination),
                      static_cast<std::streamsize>(record.payload_bytes()));
            SA_THROW_IF(!*in_,
                        "the checkpoint ended while reading '{}'s {} payload bytes",
                        record.name, record.payload_bytes());
        }

    } // namespace IO
} // namespace SushiAI
