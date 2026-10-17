/**************************************************************************/
/* cifar.cpp                                                              */
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
 * @file cifar.cpp
 * @brief Implements the reader of the CIFAR-10 and CIFAR-100 binary batch files.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/datasets/cifar.hpp>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>

#include <SushiRuntime/core/span.hpp>

#include "disk.hpp"

namespace SushiAI::Vision::Datasets
{
    namespace
    {
        /** @brief Names this reader in a message. */
        constexpr std::string_view K_READER = "read_cifar";

        /** @brief Names the training split. */
        constexpr std::string_view K_SPLIT_TRAIN = "train";

        /** @brief Names the test split. */
        constexpr std::string_view K_SPLIT_TEST = "test";

        /** @brief Holds the byte count of the three planes of one image. */
        constexpr std::uint64_t K_PIXEL_BYTES =
            static_cast<std::uint64_t>(K_CIFAR_SIDE) * K_CIFAR_SIDE * Imaging::K_RGB_CHANNELS;

        /** @brief Lists the training files of CIFAR-10. */
        constexpr std::string_view K_CIFAR10_TRAIN[] = {
            "data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin",
            "data_batch_4.bin", "data_batch_5.bin",
        };

        /** @brief Lists the test files of CIFAR-10. */
        constexpr std::string_view K_CIFAR10_TEST[] = {"test_batch.bin"};

        /** @brief Lists the training files of CIFAR-100. */
        constexpr std::string_view K_CIFAR100_TRAIN[] = {"train.bin"};

        /** @brief Lists the test files of CIFAR-100. */
        constexpr std::string_view K_CIFAR100_TEST[] = {"test.bin"};

        /** @brief Names a list of file names. */
        using FileNames = SushiRuntime::span<const std::string_view>;

        /** @brief Holds the record form and the files of one variant. */
        struct Format
        {
            /** @brief Names the variant the row describes. */
            CIFARVariant variant;

            /** @brief Counts the label bytes that open a record. */
            std::uint64_t label_bytes;

            /** @brief Holds the index, in the record, of the label byte a sample takes. */
            std::uint64_t taken_byte;

            /** @brief Counts the classes of the label taken. */
            std::int32_t class_count;

            /** @brief Lists the files of the training split. */
            FileNames train;

            /** @brief Lists the files of the test split. */
            FileNames test;
        };

        /** @brief Lists the record form and the files of every variant. */
        const Format K_FORMATS[] = {
            {CIFARVariant::CIFAR10, 1, 0, 10,
             FileNames(K_CIFAR10_TRAIN, std::size(K_CIFAR10_TRAIN)),
             FileNames(K_CIFAR10_TEST, std::size(K_CIFAR10_TEST))},
            {CIFARVariant::CIFAR100Fine, 2, 1, 100,
             FileNames(K_CIFAR100_TRAIN, std::size(K_CIFAR100_TRAIN)),
             FileNames(K_CIFAR100_TEST, std::size(K_CIFAR100_TEST))},
            {CIFARVariant::CIFAR100Coarse, 2, 0, 20,
             FileNames(K_CIFAR100_TRAIN, std::size(K_CIFAR100_TRAIN)),
             FileNames(K_CIFAR100_TEST, std::size(K_CIFAR100_TEST))},
        };

        /**
         * @brief Appends the records of one file to a list.
         * @return Success; FileNotFound for an absent file; InvalidFile for a file that is
         *         empty or not a whole number of records, and for a label at or above the
         *         class count, with the one-based record.
         */
        [[nodiscard]] Status append_file(const std::filesystem::path& file, const Format& format,
                                         SampleList& list)
        {
            const std::string name = Detail::display_of(file);
            const std::uint64_t record_bytes = format.label_bytes + K_PIXEL_BYTES;

            SA_TRY_ASSIGN(const std::uint64_t file_bytes, Detail::size_of_file(file, K_READER));
            SA_FAIL_IF(file_bytes == 0 || file_bytes % record_bytes != 0, StatusCode::InvalidFile,
                       "'{}': {}: {} bytes are not a whole number of records of {} bytes", name,
                       K_READER, file_bytes, record_bytes);

            std::ifstream in(file, std::ios::binary);
            SA_FAIL_IF(!in.is_open(), StatusCode::InvalidFile,
                       "'{}': {}: the file cannot be opened for reading", name, K_READER);

            const std::uint64_t records = file_bytes / record_bytes;
            const std::uint32_t file_index = static_cast<std::uint32_t>(list.files.size());
            list.files.push_back(file);
            list.samples.reserve(list.samples.size() + static_cast<std::size_t>(records));

            for (std::uint64_t record = 0; record < records; ++record)
            {
                const std::uint64_t start = record * record_bytes;
                char byte = 0;

                in.seekg(static_cast<std::streamoff>(start + format.taken_byte), std::ios::beg);
                in.read(&byte, 1);
                SA_FAIL_IF(in.gcount() != 1, StatusCode::InvalidFile,
                           "'{}': {}: record {} of {}: the label cannot be read", name, K_READER,
                           record + 1, records);

                const std::int32_t label = static_cast<unsigned char>(byte);
                SA_FAIL_IF(label >= format.class_count, StatusCode::InvalidFile,
                           "'{}': {}: record {} of {}: the label {} is not below the {} classes",
                           name, K_READER, record + 1, records, label, format.class_count);

                Sample sample;
                sample.file = file_index;
                sample.class_index = label;
                sample.stored = StoredPixels{start + format.label_bytes,
                                             Imaging::ImageSize{K_CIFAR_SIDE, K_CIFAR_SIDE},
                                             Imaging::K_RGB_CHANNELS, PlaneLayout::Planar};
                list.samples.push_back(std::move(sample));
            }

            return Status();
        }

        /** @brief Lists one split; may throw what an allocation or a path throws. */
        [[nodiscard]] Result<SampleList> read_checked(const CIFAROptions& options)
        {
            SA_FAIL_IF(options.split != K_SPLIT_TRAIN && options.split != K_SPLIT_TEST,
                       StatusCode::InvalidDescription,
                       "{}: the split is '{}'; it must be '{}' or '{}'", K_READER, options.split,
                       K_SPLIT_TRAIN, K_SPLIT_TEST);

            SampleList list;
            for (const Format& format : K_FORMATS)
            {
                if (format.variant != options.variant)
                    continue;

                list.class_count = format.class_count;
                for (std::string_view name : options.split == K_SPLIT_TEST ? format.test
                                                                           : format.train)
                {
                    SA_TRY(append_file(options.root / name, format, list));
                }
            }

            SA_FAIL_IF(list.samples.empty(), StatusCode::InvalidDescription,
                       "{}: the variant {} is none this build reads", K_READER,
                       static_cast<int>(options.variant));

            return Result<SampleList>(std::move(list));
        }
    } // namespace

    Result<SampleList> read_cifar(const CIFAROptions& options)
    {
        return Detail::at_boundary(K_READER, [&]()
        {
            return read_checked(options);
        }, options.root);
    }
} // namespace SushiAI::Vision::Datasets
