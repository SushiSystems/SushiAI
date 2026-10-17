/**************************************************************************/
/* reader_table.cpp                                                       */
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
 * @file reader_table.cpp
 * @brief Implements the table of the format readers a data section may name.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/datasets/reader_table.hpp>

#include <cstddef>
#include <iterator>
#include <vector>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/datasets/cifar.hpp>
#include <SushiAI/vision/datasets/coco_json.hpp>
#include <SushiAI/vision/datasets/csv.hpp>
#include <SushiAI/vision/datasets/idx.hpp>
#include <SushiAI/vision/datasets/image_folder.hpp>
#include <SushiAI/vision/datasets/mask_png.hpp>
#include <SushiAI/vision/datasets/voc_xml.hpp>
#include <SushiAI/vision/datasets/yolo_txt.hpp>

#include "disk.hpp"

namespace SushiAI::Vision::Datasets
{
    namespace
    {
        /** @brief Names this unit in a message. */
        constexpr std::string_view K_TABLE = "reader table";

        /** @brief Names the key of the annotation file of coco_json. */
        constexpr std::string_view K_ANNOTATIONS_KEY = "annotations";

        /** @brief Names the key of the image folder of coco_json and of mask_png. */
        constexpr std::string_view K_IMAGES_KEY = "images";

        /** @brief Names the key that makes coco_json keep polygons. */
        constexpr std::string_view K_POLYGONS_KEY = "polygons";

        /** @brief Names the key of the class list of voc_xml. */
        constexpr std::string_view K_CLASS_NAMES_KEY = "class_names";

        /** @brief Names the key that makes voc_xml keep objects marked as difficult. */
        constexpr std::string_view K_KEEP_DIFFICULT_KEY = "keep_difficult";

        /** @brief Names the key of the mask folder of mask_png. */
        constexpr std::string_view K_MASKS_KEY = "masks";

        /** @brief Names the key of what the stem of an image of mask_png ends in. */
        constexpr std::string_view K_IMAGE_SUFFIX_KEY = "image_suffix";

        /** @brief Names the key of what the stem of a mask of mask_png ends in. */
        constexpr std::string_view K_MASK_SUFFIX_KEY = "mask_suffix";

        /** @brief Names the key of the file of csv. */
        constexpr std::string_view K_FILE_KEY = "file";

        /** @brief Names the key of the feature columns of csv. */
        constexpr std::string_view K_FEATURES_KEY = "features";

        /** @brief Names the key of the target columns of csv. */
        constexpr std::string_view K_TARGETS_KEY = "targets";

        /** @brief Pairs a class count with the CIFAR labels that have it. */
        struct CIFARLabels
        {
            std::int32_t classes; /**< Classes the labels span. */
            CIFARVariant variant; /**< Record form and label that span them. */
        };

        /** @brief Holds the labels the cifar reader can take; the first is the default. */
        constexpr CIFARLabels K_CIFAR_LABELS[] = {
            CIFARLabels{10, CIFARVariant::CIFAR10},
            CIFARLabels{100, CIFARVariant::CIFAR100Fine},
            CIFARLabels{20, CIFARVariant::CIFAR100Coarse},
        };

        /** @brief Returns the labels of @p classes, the default for 0, or null for neither. */
        [[nodiscard]] const CIFARLabels* find_cifar_labels(std::int32_t classes) noexcept
        {
            if (classes == 0)
                return &K_CIFAR_LABELS[0];

            for (const CIFARLabels& labels : K_CIFAR_LABELS)
            {
                if (labels.classes == classes)
                    return &labels;
            }

            return nullptr;
        }

        /** @brief Reads a split of images with one label text file per image. */
        [[nodiscard]] Result<SampleList> read_yolo_txt_request(const ReaderRequest& request)
        {
            return Detail::at_boundary(K_TABLE, [&]() -> Result<SampleList>
            {
                YOLOTxtOptions options;
                options.root = request.root;
                options.split = request.split;
                options.class_count = request.class_count;
                return read_yolo_txt(options);
            });
        }

        /** @brief Reads a split kept as one folder of images per class. */
        [[nodiscard]] Result<SampleList> read_image_folder_request(const ReaderRequest& request)
        {
            return Detail::at_boundary(K_TABLE, [&]() -> Result<SampleList>
            {
                ImageFolderOptions options;
                options.root = request.root;
                options.split = request.split;
                return read_image_folder(options);
            });
        }

        /** @brief Reads a split kept as an IDX image file and an IDX label file. */
        [[nodiscard]] Result<SampleList> read_idx_request(const ReaderRequest& request)
        {
            return Detail::at_boundary(K_TABLE, [&]() -> Result<SampleList>
            {
                IDXOptions options;
                options.root = request.root;
                options.split = request.split;
                return read_idx(options);
            });
        }

        /**
         * @brief Reads a split of CIFAR binary files; the class count selects the labels.
         * @return The list of CIFAR-10 for 0 or 10 classes, of the fine CIFAR-100 labels for
         *         100 and of the coarse ones for 20; InvalidDescription for another count.
         */
        [[nodiscard]] Result<SampleList> read_cifar_request(const ReaderRequest& request)
        {
            const CIFARLabels* const labels = find_cifar_labels(request.class_count);
            SA_FAIL_IF(labels == nullptr, StatusCode::InvalidDescription,
                       "the cifar reader has no labels of {} classes; it reads 10 (CIFAR-10), "
                       "100 (CIFAR-100, fine) or 20 (CIFAR-100, coarse)",
                       request.class_count);

            return Detail::at_boundary(K_TABLE, [&]() -> Result<SampleList>
            {
                CIFAROptions options;
                options.root = request.root;
                options.split = request.split;
                options.variant = labels->variant;
                return read_cifar(options);
            });
        }

        /** @brief Returns "<context> '<key>'", what a diagnostic calls a value of a request. */
        [[nodiscard]] std::string named(const ReaderRequest& request, std::string_view key)
        {
            return request.context + " '" + std::string(key) + "'";
        }

        /**
         * @brief Returns the string a request's section holds under a key; none when absent.
         * @throws Error If the value is no string.
         */
        [[nodiscard]] std::string text_of(const ReaderRequest& request, std::string_view key)
        {
            const Config::Value absent = Config::Value::string(std::string());
            return request.section.at_or(key, absent).as_string(named(request, key));
        }

        /**
         * @brief Returns the boolean a request's section holds under a key; false when absent.
         * @throws Error If the value is no boolean.
         */
        [[nodiscard]] bool flag_of(const ReaderRequest& request, std::string_view key)
        {
            const Config::Value absent = Config::Value::boolean(false);
            return request.section.at_or(key, absent).as_bool(named(request, key));
        }

        /**
         * @brief Returns the strings a request's section lists under a key; none when absent.
         * @throws Error If the value is no array of strings.
         */
        [[nodiscard]] std::vector<std::string> names_of(const ReaderRequest& request,
                                                        std::string_view key)
        {
            const Config::Value absent = Config::Value::array({});
            const std::string name = named(request, key);
            std::vector<std::string> names;

            for (const Config::Value& entry : request.section.at_or(key, absent).as_array(name))
                names.push_back(entry.as_string(name));

            return names;
        }

        /** @brief Lists the keys the coco_json row reads. */
        constexpr std::string_view K_COCO_JSON_KEYS[] = {K_ANNOTATIONS_KEY, K_IMAGES_KEY,
                                                         K_POLYGONS_KEY};

        /** @brief Lists the keys the voc_xml row reads. */
        constexpr std::string_view K_VOC_XML_KEYS[] = {K_CLASS_NAMES_KEY, K_KEEP_DIFFICULT_KEY};

        /** @brief Lists the keys the mask_png row reads. */
        constexpr std::string_view K_MASK_PNG_KEYS[] = {K_IMAGES_KEY, K_MASKS_KEY,
                                                       K_IMAGE_SUFFIX_KEY, K_MASK_SUFFIX_KEY};

        /** @brief Lists the keys the csv row reads. */
        constexpr std::string_view K_CSV_KEYS[] = {K_FILE_KEY, K_FEATURES_KEY, K_TARGETS_KEY};

        /**
         * @brief Reads one annotation JSON and its image folder.
         * @throws Error If "annotations" or "images" is no string, or "polygons" no boolean.
         */
        [[nodiscard]] Result<SampleList> read_coco_json_request(const ReaderRequest& request)
        {
            COCOJSONOptions options;
            options.root = request.root;
            options.split = request.split;
            options.annotation_file = text_of(request, K_ANNOTATIONS_KEY);
            options.image_folder = text_of(request, K_IMAGES_KEY);
            options.segmentation = flag_of(request, K_POLYGONS_KEY) ? COCOSegmentation::Polygons
                                                                    : COCOSegmentation::Ignored;

            return Detail::at_boundary(K_TABLE, [&]() -> Result<SampleList>
            {
                return read_coco_json(options);
            });
        }

        /**
         * @brief Reads a split of images with one XML annotation per image.
         * @throws Error If "class_names" is no array of strings or "keep_difficult" no boolean.
         */
        [[nodiscard]] Result<SampleList> read_voc_xml_request(const ReaderRequest& request)
        {
            VOCXMLOptions options;
            options.root = request.root;
            options.split = request.split;
            options.class_names = names_of(request, K_CLASS_NAMES_KEY);
            options.difficult = flag_of(request, K_KEEP_DIFFICULT_KEY) ? VOCDifficult::Kept
                                                                       : VOCDifficult::Dropped;

            return Detail::at_boundary(K_TABLE, [&]() -> Result<SampleList>
            {
                return read_voc_xml(options);
            });
        }

        /**
         * @brief Reads a split of images with one PNG mask per image; the class count is read.
         * @throws Error If "images", "masks", "image_suffix" or "mask_suffix" is no string.
         */
        [[nodiscard]] Result<SampleList> read_mask_png_request(const ReaderRequest& request)
        {
            const std::string images = text_of(request, K_IMAGES_KEY);
            const std::string masks = text_of(request, K_MASKS_KEY);

            MaskPNGOptions options;
            options.root = request.root;
            options.split = request.split;
            options.class_count = request.class_count;
            options.image_suffix = text_of(request, K_IMAGE_SUFFIX_KEY);
            options.mask_suffix = text_of(request, K_MASK_SUFFIX_KEY);
            if (!images.empty())
                options.image_folder = images;
            if (!masks.empty())
                options.mask_folder = masks;

            return Detail::at_boundary(K_TABLE, [&]() -> Result<SampleList>
            {
                return read_mask_png(options);
            });
        }

        /**
         * @brief Reads the feature and the target columns of one CSV file.
         * @throws Error If "file" is no string, or "features" or "targets" no array of strings.
         */
        [[nodiscard]] Result<SampleList> read_csv_request(const ReaderRequest& request)
        {
            CSVOptions options;
            options.root = request.root;
            options.split = request.split;
            options.file = text_of(request, K_FILE_KEY);
            options.features = names_of(request, K_FEATURES_KEY);
            options.targets = names_of(request, K_TARGETS_KEY);

            return Detail::at_boundary(K_TABLE, [&]() -> Result<SampleList>
            {
                return read_csv(options);
            });
        }

        /** @brief Returns the keys of a row as the span a ReaderEntry holds. */
        template <std::size_t Count>
        [[nodiscard]] SushiRuntime::span<const std::string_view> keys_of(
            const std::string_view (&keys)[Count]) noexcept
        {
            return SushiRuntime::span<const std::string_view>(keys, Count);
        }
    } // namespace

    SushiRuntime::span<const ReaderEntry> reader_table() noexcept
    {
        // The rows hold spans, so they are built on the first call and not at compile time.
        static const ReaderEntry K_READERS[] = {
            ReaderEntry{"yolo_txt", &read_yolo_txt_request, {}},
            ReaderEntry{"image_folder", &read_image_folder_request, {}},
            ReaderEntry{"idx", &read_idx_request, {}},
            ReaderEntry{"cifar", &read_cifar_request, {}},
            ReaderEntry{"coco_json", &read_coco_json_request, keys_of(K_COCO_JSON_KEYS)},
            ReaderEntry{"voc_xml", &read_voc_xml_request, keys_of(K_VOC_XML_KEYS)},
            ReaderEntry{"mask_png", &read_mask_png_request, keys_of(K_MASK_PNG_KEYS)},
            ReaderEntry{"csv", &read_csv_request, keys_of(K_CSV_KEYS)},
        };

        return SushiRuntime::span<const ReaderEntry>(K_READERS, std::size(K_READERS));
    }

    const ReaderEntry* find_reader(std::string_view name) noexcept
    {
        for (const ReaderEntry& row : reader_table())
        {
            if (row.name == name)
                return &row;
        }

        return nullptr;
    }
} // namespace SushiAI::Vision::Datasets
