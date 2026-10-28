/**************************************************************************/
/* json_events.cpp                                                        */
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
 * @file json_events.cpp
 * @brief Implements the reader that streams a JSON file to a handler of its events.
 * @author Mustafa Garip
 */

#include <SushiAI/config/json_events.hpp>

#include <cstdint>
#include <fstream>
#include <ios>
#include <limits>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <SushiAI/seam/seam.hpp>

namespace SushiAI
{
    namespace Config
    {
        namespace
        {
            /** @brief Names the document type whose scalar types the parser hands over. */
            using Document = nlohmann::json;

            /**
             * @class EventAdapter
             * @brief Hands each event of the parser to a JSONEvents and keeps why a read ended.
             */
            class EventAdapter final
            {
                public:
                    /** @brief Binds the handler, which must outlive the adapter. */
                    explicit EventAdapter(JSONEvents& handler) noexcept : handler_(&handler)
                    {

                    }

                    /** @brief Forwards a null. */
                    bool null()
                    {
                        return handler_->null();
                    }

                    /** @brief Forwards true or false. */
                    bool boolean(bool value)
                    {
                        return handler_->boolean(value);
                    }

                    /** @brief Forwards a negative whole number. */
                    bool number_integer(Document::number_integer_t value)
                    {
                        return handler_->whole(value);
                    }

                    /** @brief Forwards a whole number, as a real above the largest int64. */
                    bool number_unsigned(Document::number_unsigned_t value)
                    {
                        constexpr Document::number_unsigned_t limit =
                            static_cast<Document::number_unsigned_t>(
                                std::numeric_limits<std::int64_t>::max());
                        if (value > limit)
                            return handler_->real(static_cast<double>(value));

                        return handler_->whole(static_cast<std::int64_t>(value));
                    }

                    /** @brief Forwards a number with a fraction or an exponent. */
                    bool number_float(Document::number_float_t value, const Document::string_t&)
                    {
                        return handler_->real(value);
                    }

                    /** @brief Forwards a string. */
                    bool string(Document::string_t& text)
                    {
                        return handler_->string(text);
                    }

                    /** @brief Refuses a byte string, which JSON text cannot spell. */
                    bool binary(Document::binary_t&)
                    {
                        return false;
                    }

                    /** @brief Forwards the opening of an object. */
                    bool start_object(std::size_t)
                    {
                        return handler_->begin_object();
                    }

                    /** @brief Forwards a key. */
                    bool key(Document::string_t& text)
                    {
                        return handler_->key(text);
                    }

                    /** @brief Forwards the closing of an object. */
                    bool end_object()
                    {
                        return handler_->end_object();
                    }

                    /** @brief Forwards the opening of an array. */
                    bool start_array(std::size_t)
                    {
                        return handler_->begin_array();
                    }

                    /** @brief Forwards the closing of an array. */
                    bool end_array()
                    {
                        return handler_->end_array();
                    }

                    /** @brief Keeps the byte offset and the text of a syntax error. */
                    bool parse_error(std::size_t position, const std::string&,
                                     const Document::exception& error)
                    {
                        has_error_ = true;
                        error_offset_ = position;
                        error_text_ = error.what();
                        return false;
                    }

                    /** @brief Reports whether the text was no JSON document. */
                    [[nodiscard]] bool has_error() const noexcept
                    {
                        return has_error_;
                    }

                    /** @brief Returns how many bytes were read when the error was met. */
                    [[nodiscard]] std::size_t error_offset() const noexcept
                    {
                        return error_offset_;
                    }

                    /** @brief Returns what the parser says of the error. */
                    [[nodiscard]] const std::string& error_text() const noexcept
                    {
                        return error_text_;
                    }

                private:
                    /// @brief The handler every event goes to.
                    JSONEvents* handler_ = nullptr;

                    /// @brief Whether parse_error() was called.
                    bool has_error_ = false;

                    /// @brief The bytes read when the error was met.
                    std::size_t error_offset_ = 0;

                    /// @brief The parser's own text of the error.
                    std::string error_text_;
            };

            /**
             * @brief Parses @p in to @p adapter and returns what ended the read.
             * @return InvalidFile for a syntax error, a failed read or a parser exception.
             */
            [[nodiscard]] Status parse_stream(std::ifstream& in, EventAdapter& adapter,
                                              const std::string& file)
            {
                try
                {
                    const bool finished = Document::sax_parse(in, &adapter);
                    SA_FAIL_IF(in.bad(), StatusCode::InvalidFile, "failed reading '{}'", file);
                    SA_FAIL_IF(!finished && adapter.has_error(), StatusCode::InvalidFile,
                               "'{}' is not valid JSON at byte {}: {}", file,
                               adapter.error_offset(), adapter.error_text());
                }
                catch (const Document::exception& error)
                {
                    SA_FAIL_IF(true, StatusCode::InvalidFile, "'{}' is not valid JSON: {}", file,
                               error.what());
                }
                catch (const std::ios_base::failure& error)
                {
                    SA_FAIL_IF(true, StatusCode::InvalidFile, "failed reading '{}': {}", file,
                               error.what());
                }

                return Status();
            }

        } // namespace

        Status read_json_events(const std::filesystem::path& path, JSONEvents& handler) noexcept
        {
            return Detail::guard(Detail::SeamPhase::Allocate, [&]() -> Status
            {
                const std::string file = path.string();

                // libstdc++ takes a buffer before the open only, the MSVC library after it only.
                std::vector<char> buffer(K_JSON_EVENT_BUFFER_BYTES);
                const std::streamsize buffer_size = static_cast<std::streamsize>(buffer.size());
                std::ifstream in;
                in.rdbuf()->pubsetbuf(buffer.data(), buffer_size);
                in.open(path, std::ios::binary);
                SA_FAIL_IF(!in, StatusCode::FileNotFound, "could not open '{}' for reading",
                           file);
                in.rdbuf()->pubsetbuf(buffer.data(), buffer_size);

                EventAdapter adapter(handler);
                return parse_stream(in, adapter, file);
            });
        }

    } // namespace Config
} // namespace SushiAI
