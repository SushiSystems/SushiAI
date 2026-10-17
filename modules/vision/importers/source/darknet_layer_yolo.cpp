/**************************************************************************/
/* darknet_layer_yolo.cpp                                                 */
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
 * @file darknet_layer_yolo.cpp
 * @brief Implements the translator of a [yolo] section.
 * @author Mustafa Garip
 */

#include "darknet_layer_table.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/vision/detect/anchor/anchor_size.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/detect/suppression_criterion.hpp>
#include <SushiAI/vision/importers/darknet_config.hpp>
#include <SushiAI/vision/importers/darknet_model.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        namespace Detail
        {
            namespace
            {
                using Detect::Anchor::AnchorSize;
                using Detect::Anchor::HeadCoding;
                using Detect::Anchor::HeadOptions;
                using Detect::SuppressionCriterion;

                /** @brief Lists the keys of a [yolo] section: those read, then those accepted. */
                constexpr std::string_view K_YOLO_KEYS[] = {
                    "mask",           "anchors",        "classes",           "num",
                    "scale_x_y",      "new_coords",     "nms_kind",          "beta_nms",
                    "jitter",         "random",         "resize",            "ignore_thresh",
                    "truth_thresh",   "iou_loss",       "iou_normalizer",    "cls_normalizer",
                    "obj_normalizer", "iou_thresh",     "objectness_smooth", "max_delta",
                };

                /** @brief Holds the prefix of a head's published name; its index follows. */
                constexpr std::string_view K_HEAD_PREFIX = "head.";

                /**
                 * @brief Reads the anchors of a section as fractions of the network input.
                 * @return InvalidDescription naming the line for an odd count, a side that is
                 *         not positive, or a `num` that is not the number of pairs.
                 */
                [[nodiscard]] Result<std::vector<AnchorSize>> read_anchors(
                    const DarknetSection& section, const Detect::Anchor::NetworkSize& network)
                {
                    SA_TRY_ASSIGN(const std::vector<double> pixels,
                                  read_real_list(section, "anchors"));
                    const std::int64_t line = find_option(section, "anchors")->line;
                    SA_FAIL_IF(pixels.size() % 2 != 0, StatusCode::InvalidDescription,
                               "line {}: 'anchors' holds {} numbers; it needs a width and a "
                               "height per anchor",
                               line, pixels.size());

                    const auto pairs = static_cast<std::int64_t>(pixels.size() / 2);
                    SA_TRY_ASSIGN(const std::int64_t num, read_integer(section, "num", pairs));
                    SA_FAIL_IF(num != pairs, StatusCode::InvalidDescription,
                               "line {}: 'num' is {}, and 'anchors' holds {} anchors", line, num,
                               pairs);

                    std::vector<AnchorSize> anchors;
                    anchors.reserve(pixels.size() / 2);
                    for (std::size_t i = 0; i < pixels.size(); i += 2)
                    {
                        SA_FAIL_IF(!(pixels[i] > 0.0) || !(pixels[i + 1] > 0.0),
                                   StatusCode::InvalidDescription,
                                   "line {}: anchor {} is {} by {}; both sides must be positive",
                                   line, i / 2, pixels[i], pixels[i + 1]);

                        anchors.push_back(AnchorSize{
                            static_cast<float>(pixels[i] / static_cast<double>(network.width)),
                            static_cast<float>(pixels[i + 1] /
                                               static_cast<double>(network.height))});
                    }
                    return anchors;
                }

                /** @brief Returns whether two anchor lists hold the same sizes in one order. */
                [[nodiscard]] bool are_same_anchors(const std::vector<AnchorSize>& a,
                                                    const std::vector<AnchorSize>& b) noexcept
                {
                    if (a.size() != b.size())
                        return false;

                    for (std::size_t i = 0; i < a.size(); ++i)
                    {
                        if (a[i].width != b[i].width || a[i].height != b[i].height)
                            return false;
                    }
                    return true;
                }

                /**
                 * @brief Reads the mask of a section: the anchors its head uses, slot by slot.
                 * @return InvalidDescription naming the line of an index outside the anchors.
                 */
                [[nodiscard]] Result<std::vector<std::int64_t>> read_mask(
                    const DarknetSection& section, std::size_t anchor_count)
                {
                    SA_TRY_ASSIGN(std::vector<std::int64_t> mask,
                                  read_integer_list(section, "mask"));
                    const std::int64_t line = find_option(section, "mask")->line;
                    for (std::int64_t index : mask)
                    {
                        SA_FAIL_IF(index < 0 || index >= static_cast<std::int64_t>(anchor_count),
                                   StatusCode::InvalidDescription,
                                   "line {}: 'mask' names anchor {}, and the section has anchors "
                                   "0 to {}",
                                   line, index, anchor_count - 1);
                    }
                    return mask;
                }

                /**
                 * @brief Reads the class count, the coding and the offset stretch of a head.
                 * @return InvalidDescription naming the line for an absent or non-positive
                 *         `classes`, a `new_coords` other than 0 or 1, or a stretch not above 0.
                 */
                [[nodiscard]] Result<HeadOptions> read_head_options(const DarknetSection& section)
                {
                    HeadOptions options;
                    SA_TRY_ASSIGN(options.num_classes, read_integer(section, "classes", 0));
                    SA_FAIL_IF(options.num_classes < 1, StatusCode::InvalidDescription,
                               "line {}: a [yolo] section needs 'classes' of 1 or more; got {}",
                               section.line, options.num_classes);

                    SA_TRY_ASSIGN(const std::int64_t coding,
                                  read_integer(section, "new_coords", 0));
                    SA_FAIL_IF(coding != 0 && coding != 1, StatusCode::InvalidDescription,
                               "line {}: 'new_coords' is {}; it is 0 or 1", section.line, coding);
                    options.coding = coding == 1 ? HeadCoding::Activated : HeadCoding::Logit;

                    SA_TRY_ASSIGN(const double stretch, read_real(section, "scale_x_y", 1.0));
                    SA_FAIL_IF(!(stretch > 0.0), StatusCode::InvalidDescription,
                               "line {}: 'scale_x_y' is {}; it must be above 0", section.line,
                               stretch);
                    options.scale_xy = static_cast<float>(stretch);
                    return options;
                }

                /**
                 * @brief Writes the suppression measure a section names into @p model.
                 * @return Unsupported naming the line of an `nms_kind` other than `greedynms`
                 *         and `diounms`; InvalidDescription for a malformed `beta_nms`.
                 */
                [[nodiscard]] Status read_suppression(const DarknetSection& section,
                                                      DarknetModel& model)
                {
                    SA_TRY_ASSIGN(const double beta,
                                  read_real(section, "beta_nms",
                                            Detect::K_DEFAULT_DISTANCE_EXPONENT));

                    const DarknetOption* kind = find_option(section, "nms_kind");
                    model.criterion = SuppressionCriterion::IoU;
                    model.distance_exponent = Detect::K_DEFAULT_DISTANCE_EXPONENT;
                    if (kind == nullptr)
                        return Status();

                    SA_FAIL_IF(kind->value != "greedynms" && kind->value != "diounms",
                               StatusCode::Unsupported,
                               "line {}: the nms_kind '{}' is not one this importer translates",
                               kind->line, kind->value);

                    // greedynms keeps the constant exponent; only diounms reads beta_nms.
                    model.criterion = SuppressionCriterion::DistanceIoU;
                    if (kind->value == "diounms")
                        model.distance_exponent = static_cast<float>(beta);

                    return Status();
                }

            } // namespace

            Status translate_yolo(const DarknetSection& section, TranslationProgress& progress)
            {
                SA_TRY(reject_unknown_options(section, K_YOLO_KEYS));

                DarknetModel& model = progress.model;
                const LayerValue source = previous_value(progress);
                SA_FAIL_IF(source.row == Config::K_MODEL_INPUT, StatusCode::InvalidDescription,
                           "line {}: a [yolo] section reads the layer before it, and none is",
                           section.line);

                SA_TRY_ASSIGN(std::vector<AnchorSize> anchors,
                              read_anchors(section, model.network));
                SA_FAIL_IF(!model.layout.anchors.empty() &&
                               !are_same_anchors(model.layout.anchors, anchors),
                           StatusCode::InvalidDescription,
                           "line {}: the anchors of this [yolo] section differ from those of an "
                           "earlier one; a model has one list",
                           section.line);

                SA_TRY_ASSIGN(std::vector<std::int64_t> mask, read_mask(section, anchors.size()));
                SA_TRY_ASSIGN(const HeadOptions options, read_head_options(section));

                const std::int64_t expected = static_cast<std::int64_t>(mask.size()) *
                                              (Detect::Anchor::K_FIRST_CLASS_ENTRY +
                                               options.num_classes);
                SA_FAIL_IF(source.channels != expected, StatusCode::InvalidDescription,
                           "line {}: a [yolo] section with {} anchors and {} classes reads {} "
                           "channels, and the layer before it has {}",
                           section.line, mask.size(), options.num_classes, expected,
                           source.channels);

                SA_TRY(read_suppression(section, model));

                DarknetHead head;
                head.output = std::string(K_HEAD_PREFIX).append(std::to_string(model.heads.size()));
                head.layer = current_layer(progress);
                head.options = options;

                Config::OutputSpec output;
                output.node = source.row;
                output.port = std::string(NN::K_SINGLE_PORT_NAME);
                output.name = head.output;

                model.layout.anchors = std::move(anchors);
                model.layout.head_anchors.push_back(std::move(mask));
                model.heads.push_back(std::move(head));
                model.spec.outputs.push_back(std::move(output));

                // A [yolo] section adds no row: it publishes the row of the layer before it.
                progress.layers.push_back(source);
                return Status();
            }

        } // namespace Detail
    } // namespace Vision::Importers
} // namespace SushiAI
