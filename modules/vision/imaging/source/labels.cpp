/**************************************************************************/
/* labels.cpp                                                             */
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
 * @file labels.cpp
 * @brief Implements the parser that turns one label file's text into a label set.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/imaging/labels.hpp>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <system_error>
#include <vector>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI::Vision::Imaging
{
    namespace
    {
        /** @brief Holds the longest row a file may have without being a polygon file. */
        constexpr std::size_t K_LONGEST_PLAIN_ROW_FIELDS = 6;

        /** @brief Holds the largest magnitude a field may have and still convert to float. */
        constexpr double K_LARGEST_FLOAT = std::numeric_limits<float>::max();

        /** @brief Locates one row's fields inside the flat field list of its file. */
        struct Row
        {
            /** @brief Holds the index of the row's first field. */
            std::size_t first = 0;

            /** @brief Counts the row's fields; at least 1. */
            std::size_t count = 0;

            /** @brief Holds the one-based line of the text the row is on. */
            std::size_t line = 0;
        };

        /** @brief Holds the fields of one file, row after row, and where each row starts. */
        struct RowTable
        {
            /** @brief Holds every field in file order, each a slice of the parsed text. */
            std::vector<std::string_view> fields;

            /** @brief Holds one entry per line that has a field. */
            std::vector<Row> rows;
        };

        /** @brief Holds the smallest magnitude a nonzero field may have, the least normal. */
        constexpr double K_SMALLEST_NORMAL = std::numeric_limits<double>::min();

        /** @brief Reports whether a character ends a line; bytes 0x1C to 0x1E do, as in Python. */
        [[nodiscard]] constexpr bool is_line_break(char character) noexcept
        {
            return character == '\n' || character == '\r' || character == '\v' ||
                   character == '\f' || character == '\x1c' || character == '\x1d' ||
                   character == '\x1e';
        }

        /** @brief Reports whether a character separates two fields of one line. */
        [[nodiscard]] constexpr bool is_field_gap(char character) noexcept
        {
            return character == ' ' || character == '\t' || character == '\x1f';
        }

        /** @brief Splits text into rows of fields, dropping every line that has no field. */
        [[nodiscard]] RowTable split_rows(std::string_view text)
        {
            RowTable table;
            Row row;
            row.line = 1;
            std::size_t position = 0;

            while (position < text.size())
            {
                const char character = text[position];

                if (is_line_break(character))
                {
                    if (row.count > 0)
                        table.rows.push_back(row);

                    // A carriage return and the line feed after it end one line, not two.
                    const bool is_pair = character == '\r' && position + 1 < text.size() &&
                                         text[position + 1] == '\n';
                    position += is_pair ? 2 : 1;
                    row = Row{table.fields.size(), 0, row.line + 1};
                    continue;
                }

                if (is_field_gap(character))
                {
                    ++position;
                    continue;
                }

                std::size_t end = position;
                while (end < text.size() && !is_line_break(text[end]) && !is_field_gap(text[end]))
                    ++end;

                table.fields.push_back(text.substr(position, end - position));
                ++row.count;
                position = end;
            }

            if (row.count > 0)
                table.rows.push_back(row);

            return table;
        }

        /** @brief Parses one whole field as a finite decimal number; InvalidFile otherwise. */
        [[nodiscard]] Result<float> parse_field(std::string_view field, std::size_t line)
        {
            std::string_view digits = field;

            // std::from_chars takes no plus sign; one is dropped unless a minus follows.
            const bool has_plus = digits.size() > 1 && digits[0] == '+' && digits[1] != '-';
            if (has_plus)
                digits.remove_prefix(1);

            double value = 0.0;
            const char* const end = digits.data() + digits.size();
            const std::from_chars_result result = std::from_chars(digits.data(), end, value);

            SA_FAIL_IF(result.ec != std::errc() || result.ptr != end, StatusCode::InvalidFile,
                       "parse_label_text: line {}: the field '{}' is not a number", line, field);
            SA_FAIL_IF(value != 0.0 && std::fabs(value) < K_SMALLEST_NORMAL,
                       StatusCode::InvalidFile,
                       "parse_label_text: line {}: the field '{}' is below the smallest normal "
                       "number",
                       line, field);
            SA_FAIL_IF(!std::isfinite(value) || std::fabs(value) > K_LARGEST_FLOAT,
                       StatusCode::InvalidFile,
                       "parse_label_text: line {}: the field '{}' is not a finite number", line,
                       field);

            return static_cast<float>(value);
        }

        /** @brief Reports whether any row is longer than a plain row may be. */
        [[nodiscard]] bool is_polygon_file(const std::vector<Row>& rows) noexcept
        {
            return std::any_of(rows.begin(), rows.end(), [](const Row& row)
            {
                return row.count > K_LONGEST_PLAIN_ROW_FIELDS;
            });
        }

        /** @brief Returns InvalidFile when a row's field count does not fit its file's kind. */
        Status check_field_count(const Row& row, bool is_polygon)
        {
            const std::size_t box_fields = static_cast<std::size_t>(K_BOX_ROW_FIELDS);
            const std::size_t polygon_fields = static_cast<std::size_t>(K_POLYGON_ROW_MIN_FIELDS);

            if (!is_polygon)
            {
                SA_FAIL_IF(row.count != box_fields, StatusCode::InvalidFile,
                           "parse_label_text: line {}: a box row has {} columns; this row has "
                           "{}",
                           row.line, box_fields, row.count);
                return Status();
            }

            SA_FAIL_IF(row.count == box_fields, StatusCode::InvalidFile,
                       "parse_label_text: line {}: a box row in a file of polygon rows", row.line);
            SA_FAIL_IF(row.count < polygon_fields || row.count % 2 == 0,
                       StatusCode::InvalidFile,
                       "parse_label_text: line {}: a polygon row has a class and at least three "
                       "points, an odd count of at least {} columns; this row has {}",
                       row.line, polygon_fields, row.count);
            return Status();
        }

        /**
         * @brief Returns the box a row states, or the box around the points of a polygon row.
         * @param values The row's parsed fields, class first; the count fits the file's kind.
         */
        [[nodiscard]] NormalizedBox box_of_row(const float* values, std::size_t count,
                                               bool is_polygon) noexcept
        {
            if (!is_polygon)
                return NormalizedBox{values[1], values[2], values[3], values[4]};

            float min_x = values[1];
            float max_x = values[1];
            float min_y = values[2];
            float max_y = values[2];

            for (std::size_t i = 3; i + 1 < count; i += 2)
            {
                min_x = std::min(min_x, values[i]);
                max_x = std::max(max_x, values[i]);
                min_y = std::min(min_y, values[i + 1]);
                max_y = std::max(max_y, values[i + 1]);
            }

            NormalizedBox box;
            box.x_center = (min_x + max_x) / 2.0f;
            box.y_center = (min_y + max_y) / 2.0f;
            box.width = max_x - min_x;
            box.height = max_y - min_y;
            return box;
        }

        /**
         * @brief Returns the points of a polygon row in file order.
         * @param values The row's parsed fields, class first; count is odd and at least 7.
         */
        [[nodiscard]] Polygon polygon_of_row(const float* values, std::size_t count)
        {
            Polygon polygon;
            polygon.reserve((count - 1) / 2);

            for (std::size_t i = 1; i + 1 < count; i += 2)
                polygon.push_back(Point{values[i], values[i + 1]});

            return polygon;
        }

        /** @brief Returns InvalidFile when one of a box's four values is outside the tolerance. */
        Status check_box(const NormalizedBox& box, std::size_t line)
        {
            const float lower = -K_LABEL_COORDINATE_SLACK;
            const float upper = 1.0f + K_LABEL_COORDINATE_SLACK;

            for (float value : {box.x_center, box.y_center, box.width, box.height})
            {
                SA_FAIL_IF(value > upper, StatusCode::InvalidFile,
                           "parse_label_text: line {}: the coordinate {} is above {}; "
                           "coordinates are fractions of the image",
                           line, value, upper);
                SA_FAIL_IF(value < lower, StatusCode::InvalidFile,
                           "parse_label_text: line {}: the coordinate {} is negative", line,
                           value);
            }
            return Status();
        }

        /** @brief Returns a row's class id; InvalidFile unless it is a whole number in range. */
        [[nodiscard]] Result<std::int32_t> class_of_row(float value, std::int32_t class_count,
                                                        std::size_t line)
        {
            SA_FAIL_IF(value < -K_LABEL_COORDINATE_SLACK, StatusCode::InvalidFile,
                       "parse_label_text: line {}: the class {} is negative", line, value);
            SA_FAIL_IF(std::trunc(value) != value, StatusCode::InvalidFile,
                       "parse_label_text: line {}: the class {} is not a whole number", line,
                       value);
            SA_FAIL_IF(static_cast<double>(value) >= static_cast<double>(class_count),
                       StatusCode::InvalidFile,
                       "parse_label_text: line {}: the class {} is out of range; the dataset "
                       "has {} classes",
                       line, value, class_count);

            return static_cast<std::int32_t>(value);
        }

        /** @brief Reports whether two rows have the same field count and equal values. */
        [[nodiscard]] bool is_same_row(const std::vector<float>& values, const Row& a,
                                       const Row& b) noexcept
        {
            const auto first = values.begin() + static_cast<std::ptrdiff_t>(a.first);
            const auto other = values.begin() + static_cast<std::ptrdiff_t>(b.first);

            return a.count == b.count &&
                   std::equal(first, first + static_cast<std::ptrdiff_t>(a.count), other);
        }
    } // namespace

    Result<FileLabels> parse_label_text(std::string_view text, std::int32_t class_count)
    {
        const RowTable table = split_rows(text);
        const bool is_polygon = is_polygon_file(table.rows);

        std::vector<float> values;
        values.reserve(table.fields.size());

        for (const Row& row : table.rows)
        {
            SA_TRY(check_field_count(row, is_polygon));

            for (std::size_t i = 0; i < row.count; ++i)
            {
                SA_TRY_ASSIGN(const float parsed,
                              parse_field(table.fields[row.first + i], row.line));
                values.push_back(parsed);
            }
        }

        FileLabels labels;
        labels.reserve(table.rows.size());

        std::vector<Row> kept;
        kept.reserve(table.rows.size());

        for (const Row& row : table.rows)
        {
            const float* const fields = values.data() + row.first;
            SA_TRY_ASSIGN(const std::int32_t class_id,
                          class_of_row(fields[0], class_count, row.line));
            const NormalizedBox box = box_of_row(fields, row.count, is_polygon);
            SA_TRY(check_box(box, row.line));

            const bool is_repeat = std::any_of(kept.begin(), kept.end(), [&](const Row& earlier)
            {
                return is_same_row(values, earlier, row);
            });
            if (is_repeat)
                continue;

            kept.push_back(row);
            labels.add(class_id, box, is_polygon ? polygon_of_row(fields, row.count) : Polygon{});
        }

        return Result<FileLabels>(std::move(labels));
    }
} // namespace SushiAI::Vision::Imaging
