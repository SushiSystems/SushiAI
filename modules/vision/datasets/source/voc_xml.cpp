/**************************************************************************/
/* voc_xml.cpp                                                            */
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
 * @file voc_xml.cpp
 * @brief Implements the reader of a dataset kept as images and one XML annotation per image.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/datasets/voc_xml.hpp>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <string_view>
#include <system_error>
#include <utility>

#include <SushiBLAS/core/format.hpp>

#include <SushiAI/core/status.hpp>
#include <SushiAI/vision/datasets/xml_element.hpp>
#include <SushiAI/vision/imaging/boxes.hpp>

#include "disk.hpp"

namespace SushiAI::Vision::Datasets
{
    namespace
    {
        /** @brief Names this reader in a message. */
        constexpr std::string_view K_READER = "read_voc_xml";

        /** @brief Names the folder that holds one XML file per image. */
        constexpr std::string_view K_ANNOTATION_FOLDER = "Annotations";

        /** @brief Names the folder that holds the images. */
        constexpr std::string_view K_IMAGE_FOLDER = "JPEGImages";

        /** @brief Names the folder that holds the image lists, below root. */
        constexpr std::string_view K_LIST_FOLDER = "ImageSets/Main";

        /** @brief Holds the extension of an annotation. */
        constexpr std::string_view K_XML_EXTENSION = ".xml";

        /** @brief Holds the extension of an image list. */
        constexpr std::string_view K_LIST_EXTENSION = ".txt";

        /** @brief Holds the extension of an image whose annotation names no file. */
        constexpr std::string_view K_IMAGE_EXTENSION = ".jpg";

        /** @brief Lists the bytes that end the stem on a line of an image list. */
        constexpr std::string_view K_LINE_PADDING = " \t\r";

        /** @brief Holds one object of an annotation before its class has an index. */
        struct VOCObject
        {
            std::string name;           /**< Names the class. */
            Imaging::NormalizedBox box; /**< Holds the box in fractions of the image. */
            std::size_t line = 0;       /**< Holds the line the object opens on. */
        };

        /** @brief Holds what one annotation gave. */
        struct VOCImage
        {
            std::filesystem::path annotation; /**< Names the XML file. */
            std::filesystem::path image;      /**< Names the image, below the image folder. */
            std::vector<VOCObject> objects;   /**< Holds the objects that were kept. */
        };

        /** @brief Returns @p stem with an extension appended, whatever dots it holds. */
        [[nodiscard]] std::filesystem::path with_extension(std::filesystem::path stem,
                                                           std::string_view extension)
        {
            stem += std::filesystem::path(extension);
            return stem;
        }

        /**
         * @brief Returns the annotations an image list names, in the order of the list.
         * @return FileNotFound for an absent list and, with the line, for a stem that has
         *         no annotation.
         */
        [[nodiscard]] Result<std::vector<std::filesystem::path>> listed_annotations(
            const VOCXMLOptions& options, const std::filesystem::path& folder)
        {
            const std::filesystem::path list =
                options.root / K_LIST_FOLDER / with_extension(options.split, K_LIST_EXTENSION);

            std::error_code ignored;
            SA_FAIL_IF(!std::filesystem::is_regular_file(list, ignored), StatusCode::FileNotFound,
                       "'{}': {}: the split '{}' has no image list '{}'",
                       Detail::display_of(options.root), K_READER, options.split,
                       Detail::display_of(list));

            SA_TRY_ASSIGN(const std::string text, Detail::read_text_file(list, K_READER));

            std::vector<std::filesystem::path> annotations;
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

                // A class list of the layout holds a flag after the stem; the stem is read.
                row = row.substr(first);
                row = row.substr(0, row.find_first_of(K_LINE_PADDING));

                std::filesystem::path annotation =
                    folder / with_extension(std::filesystem::u8path(row.begin(), row.end()),
                                            K_XML_EXTENSION);
                SA_FAIL_IF(!std::filesystem::is_regular_file(annotation, ignored),
                           StatusCode::FileNotFound,
                           "'{}': line {}: {}: the annotation '{}' is not a file",
                           Detail::display_of(list), line, K_READER,
                           Detail::display_of(annotation));

                annotations.push_back(std::move(annotation));
            }

            return Result<std::vector<std::filesystem::path>>(std::move(annotations));
        }

        /** @brief Returns the annotations of a split: its list, or every XML file for none. */
        [[nodiscard]] Result<std::vector<std::filesystem::path>> split_annotations(
            const VOCXMLOptions& options)
        {
            const std::filesystem::path folder = options.root / K_ANNOTATION_FOLDER;
            if (!options.split.empty())
                return listed_annotations(options, folder);

            SA_TRY_ASSIGN(std::vector<std::filesystem::path> files,
                          Detail::list_entries(folder, Detail::EntryKind::File,
                                               Detail::ListDepth::Children, K_READER));

            files.erase(std::remove_if(files.begin(), files.end(),
                                       [](const std::filesystem::path& file)
            {
                return file.extension() != std::filesystem::path(K_XML_EXTENSION);
            }), files.end());

            return Result<std::vector<std::filesystem::path>>(std::move(files));
        }

        /** @brief Returns the child of an element by name; InvalidFile, with the line, for none. */
        [[nodiscard]] Result<const XMLElement*> child_of(const XMLElement& parent,
                                                         std::string_view name,
                                                         const std::string& file)
        {
            const XMLElement* const child = parent.find_child(name);

            SA_FAIL_IF(child == nullptr, StatusCode::InvalidFile,
                       "'{}': line {}: {}: the element '{}' has no '{}'", file, parent.line,
                       K_READER, parent.name, name);
            return child;
        }

        /** @brief Returns the finite number a child's text spells; InvalidFile, with the line. */
        [[nodiscard]] Result<double> number_of(const XMLElement& parent, std::string_view name,
                                               const std::string& file)
        {
            SA_TRY_ASSIGN(const XMLElement* const child, child_of(parent, name, file));

            double value = 0.0;
            const char* const begin = child->text.data();
            const char* const end = begin + child->text.size();
            const std::from_chars_result result = std::from_chars(begin, end, value);

            SA_FAIL_IF(child->text.empty() || result.ec != std::errc() || result.ptr != end ||
                           !std::isfinite(value),
                       StatusCode::InvalidFile,
                       "'{}': line {}: {}: '{}' holds '{}', which is not a number", file,
                       child->line, K_READER, name, child->text);
            return value;
        }

        /**
         * @brief Reads the box of one object as fractions of an image.
         *
         * The layout counts pixels from 1 and includes both ends, so the box spans
         * [xmin - 1, xmax] by [ymin - 1, ymax]; a start below 1 is taken as 1.
         */
        [[nodiscard]] Result<Imaging::NormalizedBox> read_box(const XMLElement& object,
                                                              double width, double height,
                                                              const std::string& file)
        {
            SA_TRY_ASSIGN(const XMLElement* const bounds, child_of(object, "bndbox", file));
            SA_TRY_ASSIGN(const double xmin, number_of(*bounds, "xmin", file));
            SA_TRY_ASSIGN(const double ymin, number_of(*bounds, "ymin", file));
            SA_TRY_ASSIGN(const double xmax, number_of(*bounds, "xmax", file));
            SA_TRY_ASSIGN(const double ymax, number_of(*bounds, "ymax", file));

            SA_FAIL_IF(xmax < xmin || ymax < ymin, StatusCode::InvalidFile,
                       "'{}': line {}: {}: the box ends at ({}, {}), before its start ({}, {})",
                       file, bounds->line, K_READER, xmax, ymax, xmin, ymin);

            const double left = std::max(xmin - 1.0, 0.0);
            const double top = std::max(ymin - 1.0, 0.0);

            Imaging::NormalizedBox box;
            box.x_center = static_cast<float>((left + xmax) / 2.0 / width);
            box.y_center = static_cast<float>((top + ymax) / 2.0 / height);
            box.width = static_cast<float>((xmax - left) / width);
            box.height = static_cast<float>((ymax - top) / height);
            return box;
        }

        /**
         * @brief Reads one annotation: the image's name, its size and its objects.
         * @return InvalidFile, with the file and the line, for what parse_xml refuses and for
         *         an element or a number the layout requires and the file lacks.
         */
        [[nodiscard]] Result<VOCImage> read_annotation(const std::filesystem::path& path,
                                                       VOCDifficult difficult)
        {
            const std::string file = Detail::display_of(path);

            SA_TRY_ASSIGN(const std::string text, Detail::read_text_file(path, K_READER));

            Result<XMLElement> parsed = parse_xml(text);
            if (!parsed.status().is_ok())
                return SA_ANNOTATE(parsed.status(), "'{}'", file);

            const XMLElement root = std::move(parsed).value();

            SA_FAIL_IF(root.name != "annotation", StatusCode::InvalidFile,
                       "'{}': line {}: {}: the root element is '{}'; an annotation opens with "
                       "'annotation'",
                       file, root.line, K_READER, root.name);

            SA_TRY_ASSIGN(const XMLElement* const size, child_of(root, "size", file));
            SA_TRY_ASSIGN(const double width, number_of(*size, "width", file));
            SA_TRY_ASSIGN(const double height, number_of(*size, "height", file));
            SA_FAIL_IF(width < 1.0 || height < 1.0, StatusCode::InvalidFile,
                       "'{}': line {}: {}: the image is {} by {} pixels", file, size->line,
                       K_READER, height, width);

            VOCImage image;
            image.annotation = path;

            const XMLElement* const named = root.find_child("filename");
            image.image = named != nullptr && !named->text.empty()
                              ? std::filesystem::u8path(named->text)
                              : with_extension(path.stem(), K_IMAGE_EXTENSION);

            for (const XMLElement& object : root.children)
            {
                if (object.name != "object")
                    continue;

                const XMLElement* const flag = object.find_child("difficult");
                if (difficult == VOCDifficult::Dropped && flag != nullptr && flag->text == "1")
                    continue;

                SA_TRY_ASSIGN(const XMLElement* const name, child_of(object, "name", file));
                SA_FAIL_IF(name->text.empty(), StatusCode::InvalidFile,
                           "'{}': line {}: {}: the object names no class", file, name->line,
                           K_READER);

                VOCObject read;
                read.name = name->text;
                read.line = object.line;
                SA_TRY_ASSIGN(read.box, read_box(object, width, height, file));
                image.objects.push_back(std::move(read));
            }

            return Result<VOCImage>(std::move(image));
        }

        /** @brief Returns the class names of the objects of a split, sorted, once each. */
        [[nodiscard]] std::vector<std::string> collect_names(const std::vector<VOCImage>& images)
        {
            std::vector<std::string> names;

            for (const VOCImage& image : images)
            {
                for (const VOCObject& object : image.objects)
                    names.push_back(object.name);
            }

            std::sort(names.begin(), names.end());
            names.erase(std::unique(names.begin(), names.end()), names.end());
            return names;
        }

        /** @brief Pairs a class name with its index, for a search by name. */
        using NamedClass = std::pair<std::string_view, std::int32_t>;

        /**
         * @brief Returns the names of a class list with their indices, sorted by name.
         * @return InvalidDescription for a list of no name and for a name it holds twice.
         */
        [[nodiscard]] Result<std::vector<NamedClass>> index_names(
            const std::vector<std::string>& names)
        {
            SA_FAIL_IF(names.empty(), StatusCode::InvalidDescription,
                       "{}: the split holds no object and the class list is empty", K_READER);

            std::vector<NamedClass> lookup;
            lookup.reserve(names.size());

            for (std::size_t index = 0; index < names.size(); ++index)
                lookup.emplace_back(names[index], static_cast<std::int32_t>(index));

            std::sort(lookup.begin(), lookup.end());

            for (std::size_t entry = 1; entry < lookup.size(); ++entry)
            {
                SA_FAIL_IF(lookup[entry].first == lookup[entry - 1].first,
                           StatusCode::InvalidDescription,
                           "{}: the class list names '{}' twice, at {} and at {}", K_READER,
                           lookup[entry].first, lookup[entry - 1].second, lookup[entry].second);
            }

            return Result<std::vector<NamedClass>>(std::move(lookup));
        }

        /** @brief Lists one split; may throw what an allocation or a path throws. */
        [[nodiscard]] Result<SampleList> read_checked(const VOCXMLOptions& options)
        {
            SA_TRY_ASSIGN(const std::vector<std::filesystem::path> annotations,
                          split_annotations(options));
            SA_FAIL_IF(annotations.empty(), StatusCode::InvalidFile,
                       "'{}': {}: the split '{}' holds no annotation",
                       Detail::display_of(options.root), K_READER, options.split);

            std::vector<VOCImage> images;
            images.reserve(annotations.size());

            for (const std::filesystem::path& annotation : annotations)
            {
                SA_TRY_ASSIGN(VOCImage image, read_annotation(annotation, options.difficult));
                images.push_back(std::move(image));
            }

            SampleList list;
            list.class_names =
                options.class_names.empty() ? collect_names(images) : options.class_names;
            list.class_count = static_cast<std::int32_t>(list.class_names.size());
            list.files.reserve(images.size());
            list.samples.reserve(images.size());

            SA_TRY_ASSIGN(const std::vector<NamedClass> lookup, index_names(list.class_names));

            for (const VOCImage& image : images)
            {
                Sample sample;
                sample.file = static_cast<std::uint32_t>(list.files.size());
                sample.objects.reserve(image.objects.size());

                for (const VOCObject& object : image.objects)
                {
                    const auto found = std::lower_bound(
                        lookup.begin(), lookup.end(), NamedClass(object.name, 0));

                    SA_FAIL_IF(found == lookup.end() || found->first != object.name,
                               StatusCode::InvalidFile,
                               "'{}': line {}: {}: the object names the class '{}', which the "
                               "class list lacks",
                               Detail::display_of(image.annotation), object.line, K_READER,
                               object.name);

                    sample.objects.add(found->second, object.box);
                }

                list.files.push_back(options.root / K_IMAGE_FOLDER / image.image);
                list.samples.push_back(std::move(sample));
            }

            return Result<SampleList>(std::move(list));
        }
    } // namespace

    Result<SampleList> read_voc_xml(const VOCXMLOptions& options)
    {
        return Detail::at_boundary(K_READER, [&]()
        {
            return read_checked(options);
        }, options.root);
    }
} // namespace SushiAI::Vision::Datasets
