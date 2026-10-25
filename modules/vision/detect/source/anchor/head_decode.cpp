/**************************************************************************/
/* head_decode.cpp                                                        */
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
 * @file head_decode.cpp
 * @brief Implements the decode of an anchor head's output under both codings.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#include <SushiAI/vision/detect/anchor/head_decode.hpp>

#include <cstddef>
#include <string>
#include <vector>

#include <SushiAI/failure/check.hpp>
#include <SushiAI/vision/detect/box_format.hpp>

#include "slot_decode.hpp"

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        namespace
        {
            using Detail::ActivatedRule;
            using Detail::LogitRule;
            using Detail::SquaredSigmoidRule;
            using Detail::SlotBox;

            /** @brief Holds the extents and the settings one decode loop reads. */
            struct DecodePlan
            {
                std::size_t images = 0;   /**< Images B. */
                std::size_t height = 0;   /**< Grid rows H. */
                std::size_t width = 0;    /**< Grid columns W. */
                std::size_t classes = 0;  /**< Class entries C. */
                Detail::SlotFrame frame;  /**< Grid and stretch of the head's slots. */
                const std::vector<AnchorSize>* anchors = nullptr;     /**< Every anchor. */
                const std::vector<std::int64_t>* slots = nullptr;     /**< The head's anchors. */
            };

            /**
             * @brief Decodes every cell and anchor slot of a head under the rule @p Rule.
             * @tparam Rule Gives `activate(value)` and `size(value, anchor)` as static functions.
             */
            template<typename Rule>
            void decode_rows(const float* head, const DecodePlan& plan, float* boxes,
                             float* scores) noexcept
            {
                const std::size_t entries =
                    static_cast<std::size_t>(K_FIRST_CLASS_ENTRY) + plan.classes;

                for (std::size_t image = 0; image < plan.images; ++image)
                {
                    for (std::size_t row = 0; row < plan.height; ++row)
                    {
                        for (std::size_t col = 0; col < plan.width; ++col)
                        {
                            for (const std::int64_t anchor_index : *plan.slots)
                            {
                                const AnchorSize& anchor =
                                    (*plan.anchors)[static_cast<std::size_t>(anchor_index)];
                                const SlotBox box = Detail::decode_slot<Rule>(
                                    head, plan.frame, row, col, anchor);

                                boxes[0] = box.x;
                                boxes[1] = box.y;
                                boxes[2] = box.width;
                                boxes[3] = box.height;

                                const float objectness = Rule::activate(head[K_OBJECTNESS_ENTRY]);
                                const float* classes = head + K_FIRST_CLASS_ENTRY;
                                for (std::size_t index = 0; index < plan.classes; ++index)
                                    scores[index] = objectness * Rule::activate(classes[index]);

                                head += entries;
                                boxes += K_BOX_VALUES;
                                scores += plan.classes;
                            }
                        }
                    }
                }
            }
        } // namespace

        void decode_head(SushiRuntime::span<const float> head, const HeadShape& shape,
                         const HeadOptions& options, const AnchorLayout& layout,
                         std::int64_t head_index, SushiRuntime::span<float> boxes,
                         SushiRuntime::span<float> scores)
        {
            const Status valid = validate_head(options, layout, head_index);
            SA_FATAL_IF(!valid.is_ok(), "decode_head was given a head that does not validate: {}",
                        std::string(valid.message()));
            SA_FATAL_IF(shape.batch < 1 || shape.height < 1 || shape.width < 1,
                        "decode_head needs positive extents; got [{}, {}, {}]", shape.batch,
                        shape.height, shape.width);

            const std::size_t rows = static_cast<std::size_t>(shape.batch) *
                                     static_cast<std::size_t>(count_head_rows(shape, layout,
                                                                              head_index));
            const std::size_t cells = static_cast<std::size_t>(shape.batch) *
                                      static_cast<std::size_t>(shape.height) *
                                      static_cast<std::size_t>(shape.width);
            const std::size_t channels =
                static_cast<std::size_t>(count_head_channels(options, layout, head_index));
            const std::size_t classes = static_cast<std::size_t>(options.num_classes);
            SA_FATAL_IF(head.size() != cells * channels || boxes.size() != rows * K_BOX_VALUES ||
                            scores.size() != rows * classes,
                        "decode_head over {} rows of {} classes needs {} head, {} box and {} "
                        "score values; got {}, {} and {}",
                        rows, classes, cells * channels, rows * K_BOX_VALUES, rows * classes,
                        head.size(), boxes.size(), scores.size());

            DecodePlan plan;
            plan.images = static_cast<std::size_t>(shape.batch);
            plan.height = static_cast<std::size_t>(shape.height);
            plan.width = static_cast<std::size_t>(shape.width);
            plan.classes = classes;
            plan.frame = Detail::make_slot_frame(plan.height, plan.width, options.scale_xy);
            plan.anchors = &layout.anchors;
            plan.slots = &layout.head_anchors[static_cast<std::size_t>(head_index)];

            switch (options.coding)
            {
                case HeadCoding::Logit:
                    if (options.size == SizeForm::SquaredSigmoid)
                        decode_rows<SquaredSigmoidRule>(head.data(), plan, boxes.data(),
                                                        scores.data());
                    else
                        decode_rows<LogitRule>(head.data(), plan, boxes.data(), scores.data());
                    return;
                case HeadCoding::Activated:
                    decode_rows<ActivatedRule>(head.data(), plan, boxes.data(), scores.data());
                    return;
            }
            SA_FATAL("decode_head was given a coding outside HeadCoding");
        }

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI
