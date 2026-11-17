/**************************************************************************/
/* decision_file.cpp                                                      */
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
 * @file decision_file.cpp
 * @brief Implements the writer and the reader of the decision file.
 * @author Mustafa Garip
 */

#include <SushiAI/language/datasets/decision_file.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <ios>
#include <string>
#include <utility>

#include "disk.hpp"
#include "little_endian.hpp"

namespace SushiAI::Language::Datasets
{
    namespace
    {
        /** @brief Names the writer in a message. */
        constexpr std::string_view K_WRITER = "write_decision_file";

        /** @brief Names the reader in a message. */
        constexpr std::string_view K_READER = "read_decision_file";

        /** @brief Names the rules of a record in a message; the writer and the reader share it. */
        constexpr std::string_view K_RECORD_RULES = "DecisionRecord";

        /** @brief Holds the offset of the format version in the header. */
        constexpr std::size_t K_VERSION_OFFSET = 4;

        /** @brief Holds the offset of the vocabulary size in the header. */
        constexpr std::size_t K_VOCABULARY_OFFSET = 8;

        /** @brief Holds the offset of the record count in the header. */
        constexpr std::size_t K_COUNT_OFFSET = 12;

        /** @brief Holds the byte count of one id and of every other word of a record. */
        constexpr std::uint64_t K_WORD_BYTES = Detail::K_WORD_BYTES;

        /** @brief Holds the byte count of the four words every record opens with. */
        constexpr std::uint64_t K_RECORD_FIXED_BYTES = 4 * K_WORD_BYTES;

        /** @brief Holds the word a record states for the type Choice. */
        constexpr std::uint32_t K_TYPE_CHOICE = 0;

        /** @brief Holds the word a record states for the type YesNo. */
        constexpr std::uint32_t K_TYPE_YES_NO = 1;

        /** @brief Holds the word a record states for the type Number. */
        constexpr std::uint32_t K_TYPE_NUMBER = 2;

        /** @brief Reports whether a vocabulary size is one a decision file may state. */
        [[nodiscard]] constexpr bool is_vocabulary_size(std::int64_t size) noexcept
        {
            return size >= 1 && size <= Text::K_MAX_VOCABULARY_SIZE;
        }

        /** @brief Fails for an empty list, a long one or an id outside [0, vocabulary_size). */
        [[nodiscard]] Status check_list(const std::vector<Text::TokenID>& ids, const char* what,
                                        std::size_t record, std::int32_t vocabulary_size,
                                        StatusCode code)
        {
            SA_FAIL_IF(ids.empty() || ids.size() > K_MAX_DECISION_LIST_TOKENS, code,
                       "{}: the {} of record {} holds {} ids; it has 1 to {}", K_RECORD_RULES,
                       what, record, ids.size(), K_MAX_DECISION_LIST_TOKENS);

            for (std::size_t i = 0; i < ids.size(); ++i)
            {
                SA_FAIL_IF(ids[i] < 0 || ids[i] >= vocabulary_size, code,
                           "{}: the id {} at position {} of the {} of record {} is outside "
                           "[0, {})",
                           K_RECORD_RULES, ids[i], i, what, record, vocabulary_size);
            }

            return Status();
        }

        /** @brief Fails unless a record keeps the rules of its type; @p code is the failure. */
        [[nodiscard]] Status check_record(const DecisionRecord& record, std::size_t index,
                                          std::int32_t vocabulary_size, StatusCode code)
        {
            SA_TRY(check_list(record.text, "text", index, vocabulary_size, code));

            const bool is_choice = record.type == DecisionType::Choice;
            SA_FAIL_IF(is_choice == record.options.empty(), code,
                       "{}: record {} holds {} options; a choice has some and no other type "
                       "has any",
                       K_RECORD_RULES, index, record.options.size());
            SA_FAIL_IF(record.options.size() > K_MAX_DECISION_OPTIONS, code,
                       "{}: record {} holds {} options; it has at most {}", K_RECORD_RULES,
                       index, record.options.size(), K_MAX_DECISION_OPTIONS);

            for (const std::vector<Text::TokenID>& option : record.options)
                SA_TRY(check_list(option, "option", index, vocabulary_size, code));

            const bool choice_is_set = record.choice != 0;
            const bool number_is_set = record.number != 0.0f;

            switch (record.type)
            {
                case DecisionType::Choice:
                    SA_FAIL_IF(record.choice < 0
                                   || static_cast<std::size_t>(record.choice)
                                          >= record.options.size(),
                               code, "{}: record {} answers option {} of {}", K_RECORD_RULES,
                               index, record.choice, record.options.size());
                    SA_FAIL_IF(record.yes || number_is_set, code,
                               "{}: the choice record {} states the answer of another type",
                               K_RECORD_RULES, index);
                    break;
                case DecisionType::YesNo:
                    SA_FAIL_IF(choice_is_set || number_is_set, code,
                               "{}: the yes-or-no record {} states the answer of another type",
                               K_RECORD_RULES, index);
                    break;
                case DecisionType::Number:
                    SA_FAIL_IF(!std::isfinite(record.number), code,
                               "{}: the number record {} answers {}; it is not finite",
                               K_RECORD_RULES, index, record.number);
                    SA_FAIL_IF(choice_is_set || record.yes, code,
                               "{}: the number record {} states the answer of another type",
                               K_RECORD_RULES, index);
                    break;
            }

            return Status();
        }

        /** @brief Appends a 32-bit word, least significant byte first. */
        void put_u32(std::string& bytes, std::uint32_t value)
        {
            std::array<char, Detail::K_WORD_BYTES> word;
            Detail::store_u32(word.data(), value);
            bytes.append(word.data(), word.size());
        }

        /** @brief Appends a 64-bit word, least significant byte first. */
        void put_u64(std::string& bytes, std::uint64_t value)
        {
            std::array<char, Detail::K_LONG_BYTES> word;
            Detail::store_u64(word.data(), value);
            bytes.append(word.data(), word.size());
        }

        /** @brief Appends the ids of a list, one word each. */
        void put_ids(std::string& bytes, const std::vector<Text::TokenID>& ids)
        {
            for (const Text::TokenID id : ids)
                put_u32(bytes, static_cast<std::uint32_t>(id));
        }

        /** @brief Returns the word that holds the answer of a record. */
        [[nodiscard]] std::uint32_t answer_word(const DecisionRecord& record) noexcept
        {
            switch (record.type)
            {
                case DecisionType::Choice:
                    return static_cast<std::uint32_t>(record.choice);
                case DecisionType::YesNo:
                    return record.yes ? 1u : 0u;
                case DecisionType::Number:
                    break;
            }

            std::uint32_t bits = 0;
            std::memcpy(&bits, &record.number, sizeof bits);
            return bits;
        }

        /** @brief Returns the word a type is written as. */
        [[nodiscard]] std::uint32_t type_word(DecisionType type) noexcept
        {
            switch (type)
            {
                case DecisionType::Choice:
                    return K_TYPE_CHOICE;
                case DecisionType::YesNo:
                    return K_TYPE_YES_NO;
                case DecisionType::Number:
                    break;
            }

            return K_TYPE_NUMBER;
        }

        /** @brief Returns the bytes of a version 1 file of checked records. */
        [[nodiscard]] std::string encode(SushiRuntime::span<const DecisionRecord> records,
                                         std::int32_t vocabulary_size)
        {
            std::string bytes(K_DECISION_FILE_MAGIC);
            put_u32(bytes, K_DECISION_FILE_VERSION);
            put_u32(bytes, static_cast<std::uint32_t>(vocabulary_size));
            put_u64(bytes, records.size());

            for (const DecisionRecord& record : records)
            {
                put_u32(bytes, type_word(record.type));
                put_u32(bytes, answer_word(record));
                put_u32(bytes, static_cast<std::uint32_t>(record.text.size()));
                put_u32(bytes, static_cast<std::uint32_t>(record.options.size()));
                put_ids(bytes, record.text);

                for (const std::vector<Text::TokenID>& option : record.options)
                {
                    put_u32(bytes, static_cast<std::uint32_t>(option.size()));
                    put_ids(bytes, option);
                }
            }

            return bytes;
        }

        /** @brief Fails for a vocabulary size or a record a decision file may not hold. */
        [[nodiscard]] Status check_write(SushiRuntime::span<const DecisionRecord> records,
                                         std::int32_t vocabulary_size)
        {
            SA_FAIL_IF(!is_vocabulary_size(vocabulary_size), StatusCode::InvalidDescription,
                       "{}: a vocabulary of {} tokens; it has 1 to {}", K_WRITER, vocabulary_size,
                       Text::K_MAX_VOCABULARY_SIZE);

            for (std::size_t i = 0; i < records.size(); ++i)
            {
                SA_TRY(check_record(records[i], i, vocabulary_size,
                                    StatusCode::InvalidDescription));
            }

            return Status();
        }

        /** @brief Writes a checked version 1 file to a stream and fails when the stream does. */
        [[nodiscard]] Status write_stream(std::ostream& out,
                                          SushiRuntime::span<const DecisionRecord> records,
                                          std::int32_t vocabulary_size)
        {
            const std::string bytes = encode(records, vocabulary_size);
            out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
            out.flush();
            SA_FAIL_IF(!out, StatusCode::InvalidFile,
                       "{}: the stream failed while {} bytes were written", K_WRITER,
                       bytes.size());

            return Status();
        }

        /** @brief Walks the bytes of a whole file, failing with the offset it stopped at. */
        class ByteCursor final
        {
            public:
                /** @brief Starts at the first byte of @p bytes. */
                explicit ByteCursor(const std::string& bytes) noexcept : bytes_(bytes)
                {

                }

                /** @brief Returns the offset of the next byte. */
                [[nodiscard]] std::uint64_t offset() const noexcept
                {
                    return offset_;
                }

                /** @brief Returns how many bytes follow the cursor. */
                [[nodiscard]] std::uint64_t remaining() const noexcept
                {
                    return bytes_.size() - offset_;
                }

                /** @brief Returns the address of the next @p count bytes, and moves past them. */
                [[nodiscard]] Result<const char*> take(std::uint64_t count, const char* what)
                {
                    SA_FAIL_IF(count > remaining(), StatusCode::InvalidFile,
                               "{}: the file ends at byte {}, inside {}", K_READER,
                               bytes_.size(), what);

                    const char* const at = bytes_.data() + offset_;
                    offset_ += count;
                    return at;
                }

                /** @brief Reads one 32-bit word. */
                [[nodiscard]] Result<std::uint32_t> word(const char* what)
                {
                    SA_TRY_ASSIGN(const char* const at, take(K_WORD_BYTES, what));
                    return Detail::load_u32(at);
                }

            private:
                /// @brief Holds the bytes of the file; not owned.
                const std::string& bytes_;

                /// @brief Holds the offset of the next byte.
                std::uint64_t offset_ = 0;
        };

        /** @brief Reads a list of @p count ids, checking that its bytes are in the file. */
        [[nodiscard]] Result<std::vector<Text::TokenID>> read_ids(ByteCursor& cursor,
                                                                  std::uint32_t count,
                                                                  const char* what)
        {
            SA_FAIL_IF(count > K_MAX_DECISION_LIST_TOKENS, StatusCode::InvalidFile,
                       "{}: byte {} states {} ids in the {}; it has at most {}", K_READER,
                       cursor.offset(), count, what, K_MAX_DECISION_LIST_TOKENS);

            SA_TRY_ASSIGN(const char* const at, cursor.take(count * K_WORD_BYTES, what));

            std::vector<Text::TokenID> ids(count);
            for (std::uint32_t i = 0; i < count; ++i)
                ids[i] = static_cast<Text::TokenID>(Detail::load_u32(at + i * K_WORD_BYTES));

            return ids;
        }

        /** @brief Reads the record at the cursor and checks it. */
        [[nodiscard]] Result<DecisionRecord> read_record(ByteCursor& cursor, std::size_t index,
                                                         std::int32_t vocabulary_size)
        {
            const std::uint64_t start = cursor.offset();
            SA_TRY_ASSIGN(const std::uint32_t type, cursor.word("a record"));
            SA_TRY_ASSIGN(const std::uint32_t answer, cursor.word("a record"));
            SA_TRY_ASSIGN(const std::uint32_t text_count, cursor.word("a record"));
            SA_TRY_ASSIGN(const std::uint32_t option_count, cursor.word("a record"));

            SA_FAIL_IF(type > K_TYPE_NUMBER, StatusCode::InvalidFile,
                       "{}: byte {} states the record type {}; the types are 0 to {}", K_READER,
                       start, type, K_TYPE_NUMBER);
            SA_FAIL_IF(type == K_TYPE_YES_NO && answer > 1, StatusCode::InvalidFile,
                       "{}: byte {} states the yes-or-no answer {}; it is 0 or 1", K_READER,
                       start + K_WORD_BYTES, answer);

            DecisionRecord record;
            if (type == K_TYPE_CHOICE)
            {
                record.type = DecisionType::Choice;
                record.choice = static_cast<std::int32_t>(answer);
            }
            else if (type == K_TYPE_YES_NO)
            {
                record.type = DecisionType::YesNo;
                record.yes = answer == 1;
            }
            else
            {
                record.type = DecisionType::Number;
                std::memcpy(&record.number, &answer, sizeof answer);
            }

            SA_TRY_ASSIGN(record.text, read_ids(cursor, text_count, "text"));

            // Each option takes at least its length word, so more options than words are no file.
            SA_FAIL_IF(option_count > K_MAX_DECISION_OPTIONS
                           || option_count > cursor.remaining() / K_WORD_BYTES,
                       StatusCode::InvalidFile,
                       "{}: byte {} states {} options; the file holds fewer", K_READER,
                       start + 3 * K_WORD_BYTES, option_count);

            record.options.reserve(option_count);
            for (std::uint32_t i = 0; i < option_count; ++i)
            {
                SA_TRY_ASSIGN(const std::uint32_t length, cursor.word("an option"));
                SA_TRY_ASSIGN(std::vector<Text::TokenID> option,
                              read_ids(cursor, length, "option"));
                record.options.push_back(std::move(option));
            }

            SA_TRY(check_record(record, index, vocabulary_size, StatusCode::InvalidFile));
            return record;
        }

        /** @brief Parses a whole file held in memory. */
        [[nodiscard]] Result<DecisionFile> decode(const std::string& bytes)
        {
            SA_FAIL_IF(bytes.size() < K_DECISION_FILE_HEADER_BYTES, StatusCode::InvalidFile,
                       "{}: the file ends at byte {}, inside its header of {} bytes", K_READER,
                       bytes.size(), K_DECISION_FILE_HEADER_BYTES);
            SA_FAIL_IF(std::string_view(bytes.data(), K_DECISION_FILE_MAGIC.size())
                           != K_DECISION_FILE_MAGIC,
                       StatusCode::InvalidFile,
                       "{}: the file does not open with '{}'; it is no decision file", K_READER,
                       K_DECISION_FILE_MAGIC);

            const std::uint32_t version = Detail::load_u32(bytes.data() + K_VERSION_OFFSET);
            SA_FAIL_IF(version < 1 || version > K_DECISION_FILE_VERSION, StatusCode::InvalidFile,
                       "{}: byte {} states format version {}; this build reads versions 1 to {}",
                       K_READER, K_VERSION_OFFSET, version, K_DECISION_FILE_VERSION);

            const std::uint32_t vocabulary =
                Detail::load_u32(bytes.data() + K_VOCABULARY_OFFSET);
            SA_FAIL_IF(!is_vocabulary_size(vocabulary), StatusCode::InvalidFile,
                       "{}: byte {} states a vocabulary of {} tokens; it has 1 to {}", K_READER,
                       K_VOCABULARY_OFFSET, vocabulary, Text::K_MAX_VOCABULARY_SIZE);

            const std::uint64_t count = Detail::load_u64(bytes.data() + K_COUNT_OFFSET);
            const std::uint64_t most = (bytes.size() - K_DECISION_FILE_HEADER_BYTES)
                                       / K_RECORD_FIXED_BYTES;
            SA_FAIL_IF(count > most, StatusCode::InvalidFile,
                       "{}: byte {} states {} records; the file holds at most {}", K_READER,
                       K_COUNT_OFFSET, count, most);

            DecisionFile file;
            file.vocabulary_size = static_cast<std::int32_t>(vocabulary);
            file.records.reserve(count);

            ByteCursor cursor(bytes);
            SA_TRY(cursor.take(K_DECISION_FILE_HEADER_BYTES, "its header").status());

            for (std::uint64_t i = 0; i < count; ++i)
            {
                SA_TRY_ASSIGN(DecisionRecord record,
                              read_record(cursor, static_cast<std::size_t>(i),
                                          file.vocabulary_size));
                file.records.push_back(std::move(record));
            }

            SA_FAIL_IF(cursor.remaining() != 0, StatusCode::InvalidFile,
                       "{}: {} bytes follow the last record at byte {}", K_READER,
                       cursor.remaining(), cursor.offset());

            return file;
        }

        /** @brief Reads the whole stream of a file of a known byte count and parses it. */
        [[nodiscard]] Result<DecisionFile> read_stream(std::istream& in, std::uint64_t file_bytes)
        {
            std::string bytes(file_bytes, '\0');
            in.read(bytes.data(), static_cast<std::streamsize>(file_bytes));
            SA_FAIL_IF(static_cast<std::uint64_t>(in.gcount()) != file_bytes,
                       StatusCode::InvalidFile, "{}: the read ended at byte {} of {}", K_READER,
                       in.gcount(), file_bytes);

            return decode(bytes);
        }
    } // namespace

    bool operator==(const DecisionRecord& left, const DecisionRecord& right) noexcept
    {
        return left.type == right.type && left.text == right.text
               && left.options == right.options && left.choice == right.choice
               && left.yes == right.yes && left.number == right.number;
    }

    bool operator!=(const DecisionRecord& left, const DecisionRecord& right) noexcept
    {
        return !(left == right);
    }

    Status write_decision_file(std::ostream& out, SushiRuntime::span<const DecisionRecord> records,
                               std::int32_t vocabulary_size) noexcept
    {
        SA_TRY(check_write(records, vocabulary_size));

        return Detail::at_boundary(K_WRITER, [&]() -> Status
        {
            return write_stream(out, records, vocabulary_size);
        });
    }

    Status write_decision_file(const std::filesystem::path& file,
                               SushiRuntime::span<const DecisionRecord> records,
                               std::int32_t vocabulary_size) noexcept
    {
        SA_TRY(check_write(records, vocabulary_size));

        return Detail::at_boundary(K_WRITER, [&]() -> Status
        {
            std::ofstream out(file, std::ios::binary | std::ios::trunc);
            SA_FAIL_IF(!out.is_open(), StatusCode::InvalidFile,
                       "'{}': {}: the file cannot be opened for writing",
                       Detail::display_of(file), K_WRITER);

            return SA_ANNOTATE(write_stream(out, records, vocabulary_size), "'{}'",
                               Detail::display_of(file));
        }, file);
    }

    Result<DecisionFile> read_decision_file(std::istream& in, std::uint64_t file_bytes) noexcept
    {
        return Detail::at_boundary(K_READER, [&]() -> Result<DecisionFile>
        {
            return read_stream(in, file_bytes);
        });
    }

    Result<DecisionFile> read_decision_file(const std::filesystem::path& file) noexcept
    {
        return Detail::at_boundary(K_READER, [&]() -> Result<DecisionFile>
        {
            SA_TRY_ASSIGN(const std::uint64_t file_bytes,
                          Detail::size_of_file(file, K_READER));

            std::ifstream in(file, std::ios::binary);
            SA_FAIL_IF(!in.is_open(), StatusCode::InvalidFile,
                       "'{}': {}: the file cannot be opened for reading",
                       Detail::display_of(file), K_READER);

            Result<DecisionFile> decoded = read_stream(in, file_bytes);
            if (!decoded.has_value())
                return SA_ANNOTATE(decoded.status(), "'{}'", Detail::display_of(file));

            return decoded;
        }, file);
    }
} // namespace SushiAI::Language::Datasets
