/**************************************************************************/
/* decision_dataset.cpp                                                   */
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
 * @file decision_dataset.cpp
 * @brief Implements the dataset of decision records and the factory of its data block.
 * @author Mustafa Garip
 */

#include <SushiAI/language/datasets/decision_dataset.hpp>

#include <algorithm>
#include <cstring>
#include <utility>

#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/materialised_dataset.hpp>
#include <SushiAI/language/datasets/window_layout.hpp>

#include "disk.hpp"
#include "section.hpp"

namespace SushiAI::Language::Datasets
{
    namespace
    {
        /** @brief Names this unit in a message. */
        constexpr std::string_view K_DATASET = "DecisionDataset";

        /** @brief Names the key that holds the path of the decision file. */
        constexpr std::string_view K_FILE_KEY = "file";

        /** @brief Names the key that holds K, the options of a choice. */
        constexpr std::string_view K_OPTIONS_KEY = "options";

        /** @brief Names the key that holds the records of one batch. */
        constexpr std::string_view K_BATCH_SIZE_KEY = "batch_size";

        /** @brief Lists every key a "decisions" data block accepts beside its type. */
        constexpr std::string_view K_DECISIONS_KEYS[] = {K_FILE_KEY, Detail::K_WINDOW_KEY,
                                                         K_OPTIONS_KEY, K_BATCH_SIZE_KEY};

        /** @brief Holds the columns of the type mask, as an index. */
        constexpr std::size_t K_TYPE_COLUMNS = static_cast<std::size_t>(K_DECISION_TYPES);

        /** @brief Holds the column of the type mask a record of a type sets. */
        [[nodiscard]] constexpr std::size_t column_of(DecisionType type) noexcept
        {
            return static_cast<std::size_t>(type);
        }

        /** @brief Fails for extents outside the bounds a batch of decisions has. */
        [[nodiscard]] Status check_shape(const DecisionShape& shape)
        {
            SA_FAIL_IF(shape.window < 1 || shape.window > K_MAX_WINDOW_TOKENS,
                       StatusCode::InvalidDescription,
                       "{}: a window of {} positions; it has 1 to {}", K_DATASET, shape.window,
                       K_MAX_WINDOW_TOKENS);
            SA_FAIL_IF(shape.options < 1 || shape.options > K_MAX_DECISION_OPTIONS,
                       StatusCode::InvalidDescription,
                       "{}: {} options of a choice; it has 1 to {}", K_DATASET, shape.options,
                       K_MAX_DECISION_OPTIONS);
            return Status();
        }

        /** @brief Writes the values of a record into row @p row of the arrays of a batch. */
        void lay_out_row(const DecisionRecord& record, const DecisionShape& shape,
                         std::size_t row, DecisionRows& rows)
        {
            const std::size_t window = static_cast<std::size_t>(shape.window);
            const std::size_t slots = static_cast<std::size_t>(shape.options);

            std::copy(record.text.begin(), record.text.end(),
                      rows.tokens.begin() + static_cast<std::ptrdiff_t>(row * window));
            rows.lengths[row] = static_cast<float>(record.text.size());

            for (std::size_t k = 0; k < record.options.size(); ++k)
            {
                const std::vector<Text::TokenID>& option = record.options[k];
                const std::size_t slot = row * slots + k;

                std::copy(option.begin(), option.end(),
                          rows.options.begin() + static_cast<std::ptrdiff_t>(slot * window));
                rows.option_lengths[slot] = static_cast<float>(option.size());
                rows.option_valid[slot] = option.empty() ? 0.0f : 1.0f;
            }

            rows.type_mask[row * K_TYPE_COLUMNS + column_of(record.type)] = 1.0f;

            switch (record.type)
            {
                case DecisionType::Choice:
                    rows.choice[row] = record.choice;
                    break;
                case DecisionType::YesNo:
                    rows.yes_no[row] = record.yes ? 1.0f : 0.0f;
                    break;
                case DecisionType::Number:
                    rows.number[row] = record.number;
                    break;
            }
        }

        /** @brief Copies the values of a host array into the tensor of a batch. */
        template <typename Element>
        void copy_into(SushiBLAS::Tensor& tensor, const std::vector<Element>& values)
        {
            std::memcpy(tensor.data_as<Element>(), values.data(), values.size() * sizeof(Element));
        }
    } // namespace

    Status check_decision_fits(const DecisionRecord& record, const DecisionShape& shape,
                               std::size_t index) noexcept
    {
        const std::size_t window = static_cast<std::size_t>(shape.window);

        SA_FAIL_IF(record.text.size() > window, StatusCode::InvalidDescription,
                   "{}: the text of record {} holds {} ids; a row holds {}", K_DATASET, index,
                   record.text.size(), window);
        SA_FAIL_IF(record.options.size() > static_cast<std::size_t>(shape.options),
                   StatusCode::InvalidDescription,
                   "{}: record {} holds {} options; a row holds {}", K_DATASET, index,
                   record.options.size(), shape.options);

        for (const std::vector<Text::TokenID>& option : record.options)
        {
            SA_FAIL_IF(option.size() > window, StatusCode::InvalidDescription,
                       "{}: an option of record {} holds {} ids; a row holds {}", K_DATASET,
                       index, option.size(), window);
        }

        return Status();
    }

    Result<DecisionRows> lay_out_decisions(SushiRuntime::span<const DecisionRecord> records,
                                           const DecisionShape& shape) noexcept
    {
        SA_TRY(check_shape(shape));

        for (std::size_t i = 0; i < records.size(); ++i)
            SA_TRY(check_decision_fits(records[i], shape, i));

        return Detail::at_boundary(K_DATASET, [&]() -> Result<DecisionRows>
        {
            const std::size_t count = records.size();
            const std::size_t window = static_cast<std::size_t>(shape.window);
            const std::size_t slots = static_cast<std::size_t>(shape.options);

            DecisionRows rows;
            rows.tokens.assign(count * window, 0);
            rows.lengths.assign(count, 0.0f);
            rows.options.assign(count * slots * window, 0);
            rows.option_lengths.assign(count * slots, 0.0f);
            rows.option_valid.assign(count * slots, 0.0f);
            rows.choice.assign(count, 0);
            rows.yes_no.assign(count, 0.0f);
            rows.number.assign(count, 0.0f);
            rows.type_mask.assign(count * K_TYPE_COLUMNS, 0.0f);

            for (std::size_t row = 0; row < count; ++row)
                lay_out_row(records[row], shape, row, rows);

            return rows;
        });
    }

    Result<std::unique_ptr<DecisionDataset>> DecisionDataset::make(
        SushiBLAS::Engine& engine, const std::filesystem::path& file,
        DecisionDatasetOptions options) noexcept
    {
        SA_FAIL_IF(options.batch_size < 1, StatusCode::InvalidDescription,
                   "{}: '{}' has a batch size of {}; it must be positive", K_DATASET,
                   options.name, options.batch_size);
        SA_TRY(check_shape(options.shape));

        SA_TRY_ASSIGN(DecisionFile decoded, read_decision_file(file));

        for (std::size_t i = 0; i < decoded.records.size(); ++i)
        {
            SA_TRY(SA_ANNOTATE(check_decision_fits(decoded.records[i], options.shape, i),
                               "'{}'", Detail::display_of(file)));
        }

        SA_FAIL_IF(decoded.records.size() < static_cast<std::size_t>(options.batch_size),
                   StatusCode::InvalidDescription,
                   "'{}': {}: {} records, fewer than one batch of {}", Detail::display_of(file),
                   K_DATASET, decoded.records.size(), options.batch_size);

        return Detail::at_boundary(K_DATASET, [&]() -> Result<std::unique_ptr<DecisionDataset>>
        {
            return std::unique_ptr<DecisionDataset>(
                new DecisionDataset(engine, std::move(decoded), std::move(options)));
        }, file);
    }

    DecisionDataset::DecisionDataset(SushiBLAS::Engine& engine, DecisionFile file,
                                     DecisionDatasetOptions options)
        : engine_(&engine), file_(std::move(file)), options_(std::move(options))
    {
        const std::int64_t window = options_.shape.window;
        const std::int64_t slots = options_.shape.options;

        targets_ = {
            Data::TargetDescription{std::string(K_ROLE_LENGTHS), Shape{1}, DType::Float32, 0},
            Data::TargetDescription{std::string(K_ROLE_OPTIONS), Shape{slots, window},
                                    DType::Int32, 0},
            Data::TargetDescription{std::string(K_ROLE_OPTION_LENGTHS), Shape{slots},
                                    DType::Float32, 0},
            Data::TargetDescription{std::string(K_ROLE_OPTION_VALID), Shape{slots},
                                    DType::Float32, 0},
            Data::TargetDescription{std::string(K_ROLE_CHOICE), Shape{}, DType::Int32, slots},
            Data::TargetDescription{std::string(K_ROLE_YES_NO), Shape{1}, DType::Float32, 0},
            Data::TargetDescription{std::string(K_ROLE_NUMBER), Shape{1}, DType::Float32, 0},
            Data::TargetDescription{std::string(K_ROLE_TYPE_MASK), Shape{K_DECISION_TYPES},
                                    DType::Float32, 0}};
    }

    std::int64_t DecisionDataset::batch_size() const noexcept
    {
        return options_.batch_size;
    }

    Shape DecisionDataset::sample_shape() const
    {
        return Shape{options_.shape.window};
    }

    DType DecisionDataset::sample_dtype() const noexcept
    {
        return DType::Int32;
    }

    const std::vector<Data::TargetDescription>& DecisionDataset::targets() const noexcept
    {
        return targets_;
    }

    std::size_t DecisionDataset::batch_count() const noexcept
    {
        return file_.records.size() / static_cast<std::size_t>(options_.batch_size);
    }

    std::string_view DecisionDataset::name() const noexcept
    {
        return options_.name;
    }

    std::int32_t DecisionDataset::vocabulary_size() const noexcept
    {
        return file_.vocabulary_size;
    }

    Status DecisionDataset::read(std::size_t index, Data::Batch& batch) const
    {
        SA_FATAL_IF(index >= batch_count(), "{}: '{}' has {} batches; batch {} is outside them",
                    K_DATASET, options_.name, batch_count(), index);

        return read_rows(index, Data::RowRange{0, options_.batch_size}, batch);
    }

    Status DecisionDataset::read_rows(std::size_t index, Data::RowRange range,
                                      Data::Batch& batch) const
    {
        SA_TRY(Data::check_rows(*this, index, range));

        return Detail::at_boundary(K_DATASET, [&]() -> Status
        {
            const std::size_t first = index * static_cast<std::size_t>(options_.batch_size)
                                      + static_cast<std::size_t>(range.first);
            const SushiRuntime::span<const DecisionRecord> records(
                file_.records.data() + first, static_cast<std::size_t>(range.count));

            SA_TRY_ASSIGN(const DecisionRows rows, lay_out_decisions(records, options_.shape));
            SA_TRY(Data::prepare_batch(*engine_, range.count, Data::describe_input(*this),
                                       targets_, batch));

            copy_into(batch.inputs, rows.tokens);
            copy_into(batch.targets[0].values, rows.lengths);
            copy_into(batch.targets[1].values, rows.options);
            copy_into(batch.targets[2].values, rows.option_lengths);
            copy_into(batch.targets[3].values, rows.option_valid);
            copy_into(batch.targets[4].values, rows.choice);
            copy_into(batch.targets[5].values, rows.yes_no);
            copy_into(batch.targets[6].values, rows.number);
            copy_into(batch.targets[7].values, rows.type_mask);
            return Status();
        });
    }

    Result<DecisionsSection> read_decisions_section(const Config::Value& section,
                                                    std::string_view context) noexcept
    {
        const Config::SectionReader reader(section, context);
        SA_TRY(reader.check_object());
        SA_TRY(reader.check_keys(Config::KeySpan(K_DECISIONS_KEYS)));

        DecisionsSection keys;
        SA_TRY_ASSIGN(keys.file, Detail::read_text(reader, K_FILE_KEY));
        SA_TRY_ASSIGN(keys.options.shape.window, Detail::read_count(reader, Detail::K_WINDOW_KEY));
        SA_TRY_ASSIGN(keys.options.shape.options, Detail::read_count(reader, K_OPTIONS_KEY));

        if (reader.has(K_BATCH_SIZE_KEY))
        {
            SA_TRY_ASSIGN(keys.options.batch_size,
                          Detail::read_count(reader, K_BATCH_SIZE_KEY));
        }

        return keys;
    }

    Result<std::unique_ptr<Data::Dataset>> make_decisions_dataset(SushiBLAS::Engine& engine,
                                                                  const Config::Value& section,
                                                                  std::string_view context) noexcept
    {
        SA_TRY_ASSIGN(DecisionsSection keys, read_decisions_section(section, context));
        SA_TRY_ASSIGN(std::unique_ptr<DecisionDataset> dataset,
                      DecisionDataset::make(engine, std::filesystem::path(keys.file),
                                            std::move(keys.options)));

        return std::unique_ptr<Data::Dataset>(std::move(dataset));
    }
} // namespace SushiAI::Language::Datasets
