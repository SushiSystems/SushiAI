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
#include <string_view>
#include <tuple>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/conv2d.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/module_list.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/vision/blocks/conv_norm.hpp>

namespace SushiAI
{
    namespace Vision::Blocks
    {
        /** @brief Names the box branch of one level: two 3x3 Conv, then a biased 1x1 to 4. */
        using BoxBranch = NN::Sequential<Conv, Conv, NN::Conv2d>;

        /** @brief Names one stage of a class branch: a 3x3 DWConv, then a 1x1 Conv. */
        using ClassStage = NN::Sequential<DWConv, Conv>;

        /** @brief Names the class branch of one level: two stages, then a biased 1x1. */
        using ClassBranch = NN::Sequential<ClassStage, ClassStage, NN::Conv2d>;

        /**
         * @brief Maps each feature level to raw boxes and class logits, once per copy.
         *
         * The one-to-many copy comes first. Each copy appends the boxes of every level, then
         * the scores of every level, as "<copy>.level<l>.boxes" and "<copy>.level<l>.scores".
         */
        class DetectHead final : public NN::Node
        {
            public:
                /**
                 * @brief Builds both copies over the per-sample shapes of the levels.
                 * @param levels One [H, W, C] shape per level, finest first.
                 * @param classes The number of class logits per position; at least one.
                 * @param norm The epsilon and the momentum of every batch normalisation.
                 * @return The node, or InvalidDescription when @p levels is empty, a level is
                 *         not rank 3 or holds no channel, or @p classes is below one.
                 */
                [[nodiscard]] static Result<std::unique_ptr<DetectHead>> create(
                    NN::ShapeSpan levels, int64_t classes, NN::NormOptions norm);

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
                 * @brief Walks "cv2", "cv3", "one2one_cv2" and "one2one_cv3", each by level.
                 * @param prefix Dotted prefix the four names hang under.
                 */
                void walk_state(std::string_view prefix, const NN::StateVisitor& visitor) override;

                /** @brief Returns the number of levels the node was built over. */
                [[nodiscard]] std::size_t input_count() const noexcept override;

                /** @brief Lists the four groups of ports, each holding one port per level. */
                [[nodiscard]] SushiRuntime::span<const NN::Port> ports() const noexcept override;

            private:
                /** @brief Holds the branches of both copies under their checkpoint names. */
                struct Branches
                {
                    /** @brief Holds the one-to-many box branch of each level. */
                    NN::ModuleList<BoxBranch> cv2;

                    /** @brief Holds the one-to-many class branch of each level. */
                    NN::ModuleList<ClassBranch> cv3;

                    /** @brief Holds the one-to-one box branch of each level. */
                    NN::ModuleList<BoxBranch> one2one_cv2;

                    /** @brief Holds the one-to-one class branch of each level. */
                    NN::ModuleList<ClassBranch> one2one_cv3;

                    /** @brief Lists the four children in checkpoint order. */
                    static constexpr auto fields()
                    {
                        return std::tuple{
                            NN::child("cv2", &Branches::cv2),
                            NN::child("cv3", &Branches::cv3),
                            NN::child("one2one_cv2", &Branches::one2one_cv2),
                            NN::child("one2one_cv3", &Branches::one2one_cv3),
                        };
                    }
                };

                /** @brief Binds the branches and the ports create() derived from them. */
                DetectHead(Branches branches, std::vector<NN::Port> ports,
                           std::size_t level_count);

                /** @brief Holds the branches of both copies. */
                Branches branches_;

                /** @brief Holds the ports in the order forward() appends to them. */
                std::vector<NN::Port> ports_;

                /** @brief Holds how many values forward() reads. */
                std::size_t level_count_;
        };

        /**
         * @brief Builds a DetectHead node from "classes", with the optional "bn_epsilon" and
         *        "bn_momentum".
         * @param inputs One per-sample [H, W, C] shape per level, finest first.
         * @return The node, or InvalidDescription when @p inputs or a key does not fit.
         */
        [[nodiscard]] Result<std::unique_ptr<NN::Node>> make_detect_node(
            NN::ShapeSpan inputs, const Config::Value& attributes, std::string_view context);

    } // namespace Vision::Blocks
} // namespace SushiAI
