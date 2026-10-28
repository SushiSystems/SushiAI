/**************************************************************************/
/* yolo_txt.cpp                                                           */
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
 * @file yolo_txt.cpp
 * @brief Implements the reader of a dataset kept as images and one label file per image.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/datasets/yolo_txt.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/datasets/image_extensions.hpp>
#include <SushiAI/vision/imaging/labels.hpp>

#include "disk.hpp"

namespace SushiAI::Vision::Datasets
{
    namespace
    {
        /** @brief Names this reader in a message. */
        constexpr std::string_view K_READER = "read_yolo_txt";

        /** @brief Names the folder that holds the images of a dataset. */
        constexpr std::string_view K_IMAGE_FOLDER = "images";

        /** @brief Names the folder that holds the label files of a dataset. */
        constexpr std::string_view K_LABEL_FOLDER = "labels";

        /** @brief Holds the extension of a label file and of an image list. */
        constexpr std::string_view K_TEXT_EXTENSION = ".txt";

        /** @brief Lists the bytes that may surround the path on a line of an image list. */
        constexpr std::string_view K_LINE_PADDING = " \t\r";

        /**
         * @brief Reads the image paths of a list file, one per line with any text.
         * @return The paths, a relative one taken from @p root; FileNotFound, with the
         *         line, for a path that names no file.
         */
        [[nodiscard]] Result<std::vector<std::filesystem::path>> listed_images(
            const std::filesystem::path& list, const std::filesystem::path& root)
        {
            SA_TRY_ASSIGN(const std::string text, Detail::read_text_file(list, K_READER));

            std::vector<std::filesystem::path> images;
            std::size_t line = 0;

            for (std::size_t begin = 0; begin < text.size();)
            {
                const std::size_t end = std::min(text.find('\n', begin), text.size());
                std::string_view row(text.data() + begin, end - begin);
                begin = end + 1;
                ++line;

                const std::size_t first = row.find_first_not_of(K_LINE_PADDING);
                if (first == std::string_view::npos)
                    continue;

                row = row.substr(first, row.find_last_not_of(K_LINE_PADDING) - first + 1);

                std::filesystem::path image = std::filesystem::u8path(row.begin(), row.end());
                if (image.is_relative())
                    image = root / image;
                image = image.lexically_normal();

                std::error_code ignored;
                SA_FAIL_IF(!std::filesystem::is_regular_file(image, ignored),
                           StatusCode::FileNotFound,
                           "'{}': line {}: {}: the image '{}' is not a file",
                           Detail::display_of(list), line, K_READER, Detail::display_of(image));

                images.push_back(std::move(image));
            }

            return Result<std::vector<std::filesystem::path>>(std::move(images));
        }

        /** @brief Returns the files with an image extension below a folder, sorted by path. */
        [[nodiscard]] Result<std::vector<std::filesystem::path>> folder_images(
            const std::filesystem::path& folder)
        {
            SA_TRY_ASSIGN(std::vector<std::filesystem::path> files,
                          Detail::list_entries(folder, Detail::EntryKind::File,
                                               Detail::ListDepth::Tree, K_READER));

            std::vector<std::filesystem::path> images;
            images.reserve(files.size());

            for (std::filesystem::path& file : files)
            {
                if (has_image_extension(file))
                    images.push_back(std::move(file));
            }

            return Result<std::vector<std::filesystem::path>>(std::move(images));
        }

        /**
         * @brief Returns the images of a split from its list file or its image folder.
         * @return The paths; FileNotFound when the split has neither form.
         */
        [[nodiscard]] Result<std::vector<std::filesystem::path>> split_images(
            const YOLOTxtOptions& options)
        {
            const std::filesystem::path list =
                options.root / (options.split + std::string(K_TEXT_EXTENSION));
            const std::filesystem::path folder = options.root / K_IMAGE_FOLDER / options.split;

            std::error_code ignored;
            if (std::filesystem::is_regular_file(list, ignored))
                return listed_images(list, options.root);

            SA_FAIL_IF(!std::filesystem::is_directory(folder, ignored), StatusCode::FileNotFound,
                       "'{}': {}: the split '{}' is neither the image list '{}' nor the image "
                       "folder '{}'",
                       Detail::display_of(options.root), K_READER, options.split,
                       Detail::display_of(list), Detail::display_of(folder));

            return folder_images(folder);
        }

        /**
         * @brief Parses the label file of one image.
         * @return The objects, none for an absent file; InvalidFile, with the path and the
         *         line, for a file the parser rejects.
         */
        [[nodiscard]] Result<Imaging::FileLabels> read_labels(const std::filesystem::path& label,
                                                              std::int32_t class_count)
        {
            std::error_code ignored;
            if (std::filesystem::status(label, ignored).type() ==
                std::filesystem::file_type::not_found)
                return Result<Imaging::FileLabels>(Imaging::FileLabels());

            SA_TRY_ASSIGN(const std::string text, Detail::read_text_file(label, K_READER));

            Result<Imaging::FileLabels> parsed = Imaging::parse_label_text(text, class_count);
            if (!parsed.has_value())
                return SA_ANNOTATE(parsed.status(), "'{}'", Detail::display_of(label));

            return parsed;
        }

        /** @brief Lists one split; may throw what an allocation or a path throws. */
        [[nodiscard]] Result<SampleList> read_checked(const YOLOTxtOptions& options)
        {
            SA_FAIL_IF(options.class_count < 1, StatusCode::InvalidDescription,
                       "{}: the class count is {}; it must be at least 1", K_READER,
                       options.class_count);

            SA_TRY_ASSIGN(std::vector<std::filesystem::path> images, split_images(options));
            SA_FAIL_IF(images.empty(), StatusCode::InvalidFile,
                       "'{}': {}: the split '{}' holds no image", Detail::display_of(options.root),
                       K_READER, options.split);

            SampleList list;
            list.class_count = options.class_count;
            list.samples.reserve(images.size());

            for (std::size_t image = 0; image < images.size(); ++image)
            {
                SA_TRY_ASSIGN(Imaging::FileLabels objects,
                              read_labels(yolo_label_path(images[image]), options.class_count));

                Sample sample;
                sample.file = static_cast<std::uint32_t>(image);
                sample.objects = std::move(objects);
                list.samples.push_back(std::move(sample));
            }

            list.files = std::move(images);
            return Result<SampleList>(std::move(list));
        }
    } // namespace

    std::filesystem::path yolo_label_path(const std::filesystem::path& image)
    {
        const std::filesystem::path image_folder(K_IMAGE_FOLDER);
        const std::vector<std::filesystem::path> parts(image.begin(), image.end());

        // parts.size() marks a path with no "images" folder; the last part is the file name.
        std::size_t replaced = parts.size();
        for (std::size_t part = 0; part + 1 < parts.size(); ++part)
        {
            if (parts[part] == image_folder)
                replaced = part;
        }

        std::filesystem::path label;
        for (std::size_t part = 0; part < parts.size(); ++part)
        {
            if (part == replaced)
                label /= K_LABEL_FOLDER;
            else
                label /= parts[part];
        }

        label.replace_extension(K_TEXT_EXTENSION);
        return label;
    }

    Result<SampleList> read_yolo_txt(const YOLOTxtOptions& options)
    {
        return Detail::at_boundary(K_READER, [&]()
        {
            return read_checked(options);
        }, options.root);
    }
} // namespace SushiAI::Vision::Datasets
