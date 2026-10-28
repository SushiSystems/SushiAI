/**************************************************************************/
/* detect_head.hpp                                                        */
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
 * @file detect_head.hpp
 * @brief Declares DetectHead, the two copies of the box and class branches over each level.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <variant>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/conv2d.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/module_list.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/vision/blocks/conv_norm.hpp>
#include <SushiAI/vision/detect/free/head_contract.hpp>

namespace SushiAI
{
    namespace Vision::Blocks
    {
        /** @brief Names the box branch of one level: two 3x3 Conv, then a biased 1x1. */
        using BoxBranch = NN::Sequential<Conv, Conv, NN::Conv2d>;

        /** @brief Names one stage of a class branch: a 3x3 DWConv, then a 1x1 Conv. */
        using ClassStage = NN::Sequential<DWConv, Conv>;

        /** @brief Names the class branch of one level: two stages, then a biased 1x1. */
        using ClassBranch = NN::Sequential<ClassStage, ClassStage, NN::Conv2d>;

        /** @brief Names the plain class branch of one level: two 3x3 Conv, then a biased 1x1. */
        using PlainClassBranch = NN::Sequential<Conv, Conv, NN::Conv2d>;

        /** @brief Names the form of a head's class branches. */
        enum class ClassBranchStyle : std::uint8_t
        {
            Depthwise, /**< Two depthwise-style stages, then a 1x1. */
            Plain,     /**< Three layers: two 3x3 Conv, then a 1x1. */
        };

        using Detect::Free::K_DETECT_ONE_TO_MANY;
        using Detect::Free::K_DETECT_ONE_TO_ONE;
        using Detect::Free::K_DETECT_BOXES;
        using Detect::Free::K_DETECT_SCORES;
        using Detect::Free::K_DETECT_BOX_BRANCH;
        using Detect::Free::K_DETECT_CLASS_BRANCH;
        using Detect::Free::K_DETECT_ONE_TO_ONE_BOX_BRANCH;
        using Detect::Free::K_DETECT_ONE_TO_ONE_CLASS_BRANCH;
        using Detect::Free::K_DETECT_LAST_CONV_INDEX;
        using Detect::Free::K_DETECT_BOX_BIAS;
        using Detect::Free::K_DETECT_OBJECTS_PER_IMAGE;
        using Detect::Free::K_DETECT_REFERENCE_IMAGE_SIDE;
        using Detect::Free::detect_port_name;
        using Detect::Free::detect_class_bias;

        /**
         * @brief Holds the branches of both copies of a head under their checkpoint names.
         * @tparam ClassType The class branch form, ClassBranch or PlainClassBranch.
         */
        template<typename ClassType>
        struct DetectBranches
        {
            /** @brief Holds the one-to-many box branch of each level. */
            NN::ModuleList<BoxBranch> cv2;

            /** @brief Holds the one-to-many class branch of each level. */
            NN::ModuleList<ClassType> cv3;

            /** @brief Holds the one-to-one box branch of each level; empty without end2end. */
            NN::ModuleList<BoxBranch> one2one_cv2;

            /** @brief Holds the one-to-one class branch of each level; empty without end2end. */
            NN::ModuleList<ClassType> one2one_cv3;

            /** @brief Lists the four children in checkpoint order. */
            static constexpr auto fields() noexcept
            {
                return std::tuple{
                    NN::child(K_DETECT_BOX_BRANCH, &DetectBranches::cv2),
                    NN::child(K_DETECT_CLASS_BRANCH, &DetectBranches::cv3),
                    NN::child(K_DETECT_ONE_TO_ONE_BOX_BRANCH, &DetectBranches::one2one_cv2),
                    NN::child(K_DETECT_ONE_TO_ONE_CLASS_BRANCH, &DetectBranches::one2one_cv3),
                };
            }
        };

        /** @brief Holds the structural choices of a head beyond its levels and class count. */
        struct DetectOptions
        {
            /** @brief Holds the distribution bins per box side; the box branch emits 4 times it. */
            int64_t reg_max = 1;

            /** @brief Tells whether the one-to-one copy exists next to the one-to-many copy. */
            bool end2end = true;

            /** @brief Holds the form of the class branches. */
            ClassBranchStyle class_branch = ClassBranchStyle::Depthwise;

            /**
             * @brief Holds the stride of each level, finest first. Empty leaves every bias at
             *        zero; otherwise the last box convolution starts at 1 and each last class
             *        convolution at log(5 / classes / (640 / stride)^2). A row cannot state it,
             *        so only a caller that knows the input size sets it.
             */
            std::vector<double> strides{};
        };

        /**
         * @brief Maps each feature level to raw boxes and class logits, once per copy.
         *
         * The one-to-many copy comes first, then the one-to-one copy unless end2end is off. Each
         * copy appends the boxes of every level, then the scores, as "<copy>.level<l>.boxes"
         * and "<copy>.level<l>.scores".
         */
        class DetectHead final : public NN::Node
        {
            public:
                /**
                 * @brief Builds both copies over the per-sample shapes of the levels.
                 * @param levels One [H, W, C] shape per level, finest first.
                 * @param classes The number of class logits per position; at least one.
                 * @param options The box width, the copies, the class branch form, the strides.
                 * @return The node, or InvalidDescription for a bad level, classes, reg_max
                 *         or strides.
                 */
                [[nodiscard]] static Result<std::unique_ptr<DetectHead>> create(
                    NN::ShapeSpan levels, int64_t classes, NN::NormOptions norm,
                    const DetectOptions& options = DetectOptions()) noexcept;

                /**
                 * @brief Traces both copies and appends one id per port, in the order of ports().
                 *
                 * A training trace passes each feature through STOP_GRADIENT before the
                 * one-to-one copy; an evaluation trace does not.
                 * @return InvalidDescription unless @p inputs holds one value per level, each
                 *         of the width the node was built over; @p outputs is unchanged then.
                 */
                [[nodiscard]] Status forward(const NN::Trace& trace, NN::ValueSpan inputs,
                                             std::vector<Graph::ValueID>& outputs) override;

                /**
                 * @brief Walks "cv2", "cv3", "one2one_cv2" and "one2one_cv3", each by level; the
                 *        one-to-one names are absent without end2end.
                 * @param prefix Dotted prefix the four names hang under.
                 */
                void walk_state(std::string_view prefix, const NN::StateVisitor& visitor) override;

                /** @brief Returns the number of levels the node was built over. */
                [[nodiscard]] std::size_t input_count() const noexcept override;

                /** @brief Lists the four groups of ports, each holding one port per level. */
                [[nodiscard]] SushiRuntime::span<const NN::Port> ports() const noexcept override;

            private:
                /** @brief Names the branches of either class branch form. */
                using Branches = std::variant<DetectBranches<ClassBranch>,
                                              DetectBranches<PlainClassBranch>>;

                /** @brief Binds the branches and the ports create() derived from them. */
                DetectHead(Branches branches, std::vector<NN::Port> ports,
                           std::size_t level_count, bool end2end);

                /** @brief Holds the branches of both copies. */
                Branches branches_;

                /** @brief Holds the ports in the order forward() appends to them. */
                std::vector<NN::Port> ports_;

                /** @brief Holds how many values forward() reads. */
                std::size_t level_count_;

                /** @brief Holds whether forward() traces the one-to-one copy. */
                bool end2end_;
        };

        /**
         * @brief Builds a DetectHead node from "classes", with the optional "reg_max",
         *        "end2end", "class_branch" ("depthwise" or "plain"), "bn_epsilon" and
         *        "bn_momentum".
         * @param inputs One per-sample [H, W, C] shape per level, finest first.
         * @return The node, or InvalidDescription when @p inputs or a key does not fit.
         */
        [[nodiscard]] Result<std::unique_ptr<NN::Node>> make_detect_node(
            NN::ShapeSpan inputs, const Config::Value& attributes, std::string_view context) noexcept;

    } // namespace Vision::Blocks
} // namespace SushiAI
