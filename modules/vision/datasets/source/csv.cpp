/**************************************************************************/
/* csv.cpp                                                                */
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
 * @file csv.cpp
 * @brief Implements the reader of a dataset kept as a header row and numeric columns.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/datasets/csv.hpp>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>
#include <system_error>
#include <utility>

#include <SushiBLAS/core/format.hpp>

#include <SushiAI/core/status.hpp>

#include "disk.hpp"

namespace SushiAI::Vision::Datasets
{
    namespace
    {
        /** @brief Names this reader in a message. */
        constexpr std::string_view K_READER = "read_csv";

        /** @brief Holds the extension of the file a split names. */
        constexpr std::string_view K_CSV_EXTENSION = ".csv";

        /** @brief Holds the three bytes a UTF-8 file may open with. */
        constexpr std::string_view K_BYTE_ORDER_MARK = "\xEF\xBB\xBF";

        /** @brief Lists the bytes dropped around a header name and around a number. */
        constexpr std::string_view K_CELL_PADDING = " \t";

        /** @brief Holds the text of one file and the place of its next unread record. */
        struct CSVCursor
        {
            std::string_view text;    /**< Holds the whole file. */
            std::string file;         /**< Names the file in a message. */
            std::size_t position = 0; /**< Counts the bytes read. */
            std::size_t line = 1;     /**< Holds the one-based line of the next unread byte. */
        };

        /** @brief Holds the positions, in a record, of the columns a row of the table takes. */
        struct ColumnPicks
        {
            std::vector<std::size_t> features; /**< Holds the feature columns, in row order. */
            std::vector<std::size_t> targets;  /**< Holds the target columns, in row order. */
        };

        /** @brief Reports whether a byte ends a field that is not quoted. */
        [[nodiscard]] constexpr bool ends_field(char c) noexcept
        {
            return c == ',' || c == '\n' || c == '\r';
        }

        /** @brief Returns a view without the spaces and tabs around it. */
        [[nodiscard]] std::string_view trimmed(std::string_view text) noexcept
        {
            const std::size_t first = text.find_first_not_of(K_CELL_PADDING);
            if (first == std::string_view::npos)
                return std::string_view();

            return text.substr(first, text.find_last_not_of(K_CELL_PADDING) - first + 1);
        }

        /**
         * @brief Reads a quoted field after its opening quote; a doubled quote is one quote.
         * @return InvalidFile, with the line of the opening quote, for a field left open,
         *         and with the column for text after the closing quote.
         */
        [[nodiscard]] Status read_quoted_field(CSVCursor& cursor, std::size_t column,
                                               std::string& field)
        {
            const std::size_t opened = cursor.line;

            for (++cursor.position;; ++cursor.position)
            {
                SA_FAIL_IF(cursor.position >= cursor.text.size(), StatusCode::InvalidFile,
                           "'{}': line {}: column {}: {}: the quoted field is not closed",
                           cursor.file, opened, column, K_READER);

                const char c = cursor.text[cursor.position];
                const bool is_doubled = c == '"' && cursor.position + 1 < cursor.text.size() &&
                                        cursor.text[cursor.position + 1] == '"';
                if (c == '"' && !is_doubled)
                    break;

                if (is_doubled)
                    ++cursor.position;
                if (c == '\n')
                    ++cursor.line;

                field.push_back(c);
            }

            ++cursor.position;
            SA_FAIL_IF(cursor.position < cursor.text.size() &&
                           !ends_field(cursor.text[cursor.position]),
                       StatusCode::InvalidFile,
                       "'{}': line {}: column {}: {}: text follows the closing quote of the field",
                       cursor.file, cursor.line, column, K_READER);
            return Status();
        }

        /** @brief Reads a field with no quote around it; InvalidFile for a quote inside it. */
        [[nodiscard]] Status read_plain_field(CSVCursor& cursor, std::size_t column,
                                              std::string& field)
        {
            for (; cursor.position < cursor.text.size() &&
                   !ends_field(cursor.text[cursor.position]);
                 ++cursor.position)
            {
                SA_FAIL_IF(cursor.text[cursor.position] == '"', StatusCode::InvalidFile,
                           "'{}': line {}: column {}: {}: a quote stands inside a field that "
                           "does not open with one",
                           cursor.file, cursor.line, column, K_READER);

                field.push_back(cursor.text[cursor.position]);
            }

            return Status();
        }

        /**
         * @brief Reads the fields of one record and the line break that ends it.
         * @param fields Replaced by the fields; a blank line gives one field of no text.
         * @pre The cursor is not at the end of its text.
         */
        [[nodiscard]] Status read_record(CSVCursor& cursor, std::vector<std::string>& fields)
        {
            fields.clear();

            for (bool has_more = true; has_more;)
            {
                const bool is_quoted = cursor.position < cursor.text.size() &&
                                       cursor.text[cursor.position] == '"';

                fields.emplace_back();
                if (is_quoted)
                    SA_TRY(read_quoted_field(cursor, fields.size(), fields.back()));
                else
                    SA_TRY(read_plain_field(cursor, fields.size(), fields.back()));

                has_more = cursor.position < cursor.text.size() &&
                           cursor.text[cursor.position] == ',';
                if (has_more)
                    ++cursor.position;
            }

            if (cursor.position < cursor.text.size() && cursor.text[cursor.position] == '\r')
                ++cursor.position;
            if (cursor.position < cursor.text.size() && cursor.text[cursor.position] == '\n')
                ++cursor.position;

            ++cursor.line;
            return Status();
        }

        /** @brief Reports whether a record is a line with nothing on it. */
        [[nodiscard]] bool is_blank(const std::vector<std::string>& fields) noexcept
        {
            return fields.size() == 1 && fields[0].empty();
        }

        /** @brief Joins names into "'a', 'b'" for a message. */
        [[nodiscard]] std::string quoted_list(const std::vector<std::string>& names)
        {
            std::string listed;
            for (const std::string& name : names)
            {
                if (!listed.empty())
                    listed.append(", ");

                listed.append("'").append(name).append("'");
            }

            return listed;
        }

        /**
         * @brief Returns the positions of the named columns of a header.
         * @param role Says "feature" or "target" in a message.
         * @return InvalidDescription, with every name of the header, for a name it lacks.
         */
        [[nodiscard]] Result<std::vector<std::size_t>> find_columns(
            const std::vector<std::string>& header, const std::vector<std::string>& names,
            std::string_view role, const std::string& file)
        {
            std::vector<std::size_t> columns;
            columns.reserve(names.size());

            for (const std::string& name : names)
            {
                const auto found = std::find(header.begin(), header.end(), name);
                SA_FAIL_IF(found == header.end(), StatusCode::InvalidDescription,
                           "'{}': {}: the header has no {} column '{}'; it has {}", file,
                           K_READER, role, name, quoted_list(header));

                columns.push_back(static_cast<std::size_t>(found - header.begin()));
            }

            return Result<std::vector<std::size_t>>(std::move(columns));
        }

        /**
         * @brief Chooses the feature and the target columns of a header.
         * @return The picks; InvalidDescription for a name the header lacks, for a column
         *         named as both, and when no feature column is left.
         */
        [[nodiscard]] Result<ColumnPicks> pick_columns(const std::vector<std::string>& header,
                                                       const CSVOptions& options,
                                                       const std::string& file)
        {
            ColumnPicks picks;

            SA_TRY_ASSIGN(picks.targets, find_columns(header, options.targets, "target", file));
            if (options.targets.empty())
                picks.targets.push_back(header.size() - 1);

            const auto is_target = [&](std::size_t column)
            {
                return std::find(picks.targets.begin(), picks.targets.end(), column) !=
                       picks.targets.end();
            };

            SA_TRY_ASSIGN(picks.features,
                          find_columns(header, options.features, "feature", file));

            for (std::size_t column = 0; options.features.empty() && column < header.size();
                 ++column)
            {
                if (!is_target(column))
                    picks.features.push_back(column);
            }

            for (std::size_t column : picks.features)
            {
                SA_FAIL_IF(is_target(column), StatusCode::InvalidDescription,
                           "'{}': {}: the column '{}' is named as a feature and as a target",
                           file, K_READER, header[column]);
            }

            SA_FAIL_IF(picks.features.empty(), StatusCode::InvalidDescription,
                       "'{}': {}: no column is left as a feature; the header has {}", file,
                       K_READER, quoted_list(header));
            return picks;
        }

        /** @brief Returns the number a whole cell spells, or nothing when it is no finite float. */
        [[nodiscard]] std::optional<float> parse_cell(std::string_view cell) noexcept
        {
            std::string_view digits = trimmed(cell);

            // std::from_chars takes no plus sign; one is dropped unless a sign follows it.
            if (digits.size() > 1 && digits[0] == '+' && digits[1] != '-' && digits[1] != '+')
                digits.remove_prefix(1);

            double value = 0.0;
            const char* const end = digits.data() + digits.size();
            const std::from_chars_result result = std::from_chars(digits.data(), end, value);

            const bool is_number = !digits.empty() && result.ec == std::errc() &&
                                   result.ptr == end && std::isfinite(value) &&
                                   std::fabs(value) <= std::numeric_limits<float>::max();
            if (!is_number)
                return std::nullopt;

            return static_cast<float>(value);
        }

        /**
         * @brief Reads the header: the first record that is no blank line.
         * @return The names without the spaces around them; InvalidFile for a file with no
         *         record and, with both columns, for a name that occurs twice.
         */
        [[nodiscard]] Result<std::vector<std::string>> read_header(CSVCursor& cursor)
        {
            std::vector<std::string> header;
            std::size_t line = cursor.line;

            while (cursor.position < cursor.text.size() && (header.empty() || is_blank(header)))
            {
                line = cursor.line;
                SA_TRY(read_record(cursor, header));
            }

            SA_FAIL_IF(header.empty() || is_blank(header), StatusCode::InvalidFile,
                       "'{}': {}: the file holds no header row", cursor.file, K_READER);

            for (std::size_t column = 0; column < header.size(); ++column)
            {
                header[column] = std::string(trimmed(header[column]));

                const auto first = std::find(header.begin(), header.begin() + column,
                                             header[column]);
                SA_FAIL_IF(first != header.begin() + column, StatusCode::InvalidFile,
                           "'{}': line {}: column {}: {}: the name '{}' is also the name of "
                           "column {}",
                           cursor.file, line, column + 1, K_READER, header[column],
                           first - header.begin() + 1);
            }

            return Result<std::vector<std::string>>(std::move(header));
        }

        /**
         * @brief Appends the picked cells of one record to the values of a table.
         * @param line Holds the line the record starts on.
         * @return InvalidFile, with the line, the column and its name, for a cell that is
         *         no finite number.
         */
        [[nodiscard]] Status append_row(const std::vector<std::string>& fields,
                                        const std::vector<std::size_t>& columns,
                                        const std::vector<std::string>& header,
                                        const std::string& file, std::size_t line,
                                        std::vector<float>& values)
        {
            for (std::size_t column : columns)
            {
                const std::optional<float> value = parse_cell(fields[column]);

                SA_FAIL_IF(!value.has_value(), StatusCode::InvalidFile,
                           "'{}': line {}: column {} ('{}'): {}: the cell '{}' is not a finite "
                           "number",
                           file, line, column + 1, header[column], K_READER, fields[column]);

                values.push_back(*value);
            }

            return Status();
        }

        /** @brief Reads one file; may throw what an allocation or a path throws. */
        [[nodiscard]] Result<SampleList> read_checked(const CSVOptions& options,
                                                      const std::filesystem::path& path)
        {
            SA_TRY_ASSIGN(const std::string text, Detail::read_text_file(path, K_READER));

            CSVCursor cursor;
            cursor.text = text;
            cursor.file = Detail::display_of(path);
            if (cursor.text.substr(0, K_BYTE_ORDER_MARK.size()) == K_BYTE_ORDER_MARK)
                cursor.position = K_BYTE_ORDER_MARK.size();

            SA_TRY_ASSIGN(const std::vector<std::string> header, read_header(cursor));
            SA_TRY_ASSIGN(const ColumnPicks picks, pick_columns(header, options, cursor.file));

            SampleList list;
            list.files.push_back(path);
            list.table.feature_count = static_cast<std::int32_t>(picks.features.size());
            list.table.target_count = static_cast<std::int32_t>(picks.targets.size());

            for (const std::vector<std::size_t>* columns : {&picks.features, &picks.targets})
            {
                for (std::size_t column : *columns)
                    list.table.columns.push_back(header[column]);
            }

            std::vector<std::string> fields;
            while (cursor.position < cursor.text.size())
            {
                const std::size_t line = cursor.line;
                SA_TRY(read_record(cursor, fields));

                if (is_blank(fields))
                    continue;

                SA_FAIL_IF(fields.size() != header.size(), StatusCode::InvalidFile,
                           "'{}': line {}: {}: the record has {} fields; the header has {}",
                           cursor.file, line, K_READER, fields.size(), header.size());
                SA_TRY(append_row(fields, picks.features, header, cursor.file, line,
                                  list.table.values));
                SA_TRY(append_row(fields, picks.targets, header, cursor.file, line,
                                  list.table.values));

                list.samples.emplace_back();
            }

            SA_FAIL_IF(list.samples.empty(), StatusCode::InvalidFile,
                       "'{}': {}: the file holds no record below its header", cursor.file,
                       K_READER);
            return Result<SampleList>(std::move(list));
        }
    } // namespace

    Result<SampleList> read_csv(const CSVOptions& options)
    {
        return Detail::at_boundary(K_READER, [&]() -> Result<SampleList>
        {
            if (options.file.empty())
                return read_checked(options,
                                    options.root / (options.split + std::string(K_CSV_EXTENSION)));

            return read_checked(options, options.file.is_relative() ? options.root / options.file
                                                                    : options.file);
        }, options.root);
    }
} // namespace SushiAI::Vision::Datasets
