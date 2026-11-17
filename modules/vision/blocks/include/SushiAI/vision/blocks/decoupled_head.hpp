/**************************************************************************/
/* decoupled_head.hpp                                                     */
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
 * @file decoupled_head.hpp
 * @brief Declares DecoupledHead, the YOLOX head: a shared stem, then a class and a box branch.
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
#include <SushiAI/vision/detect/yolox/yolox_head_contract.hpp>

namespace SushiAI
{
    namespace Vision::Blocks
    {
        /** @brief Names the two 3x3 Conv that start a class branch or a box branch. */
        using DecoupledTower = NN::Sequential<Conv, Conv>;

        /** @brief Holds the layers of every level under the state names of the YOLOX head. */
        struct DecoupledBranches
        {
            /** @brief Holds the class tower of each level, "cls_convs.<level>.<index>". */
            NN::ModuleList<DecoupledTower> cls_convs;

            /** @brief Holds the box tower of each level, "reg_convs.<level>.<index>". */
            NN::ModuleList<DecoupledTower> reg_convs;

            /** @brief Holds the biased 1x1 that ends each class tower, "cls_preds.<level>". */
            NN::ModuleList<NN::Conv2d> cls_preds;

            /** @brief Holds the biased 1x1 that ends each box tower, "reg_preds.<level>". */
            NN::ModuleList<NN::Conv2d> reg_preds;

            /** @brief Holds the biased 1x1 that reads each box tower for the objectness. */
            NN::ModuleList<NN::Conv2d> obj_preds;

            /** @brief Holds the 1x1 Conv that narrows each level's feature, "stems.<level>". */
            NN::ModuleList<Conv> stems;

            /** @brief Lists the six children in the order of the YOLOX state dict. */
            static constexpr auto fields() noexcept
            {
                return std::tuple{
                    NN::child("cls_convs", &DecoupledBranches::cls_convs),
                    NN::child("reg_convs", &DecoupledBranches::reg_convs),
                    NN::child("cls_preds", &DecoupledBranches::cls_preds),
                    NN::child("reg_preds", &DecoupledBranches::reg_preds),
                    NN::child("obj_preds", &DecoupledBranches::obj_preds),
                    NN::child("stems", &DecoupledBranches::stems),
                };
            }
        };

        /**
         * @brief Maps each feature level to class logits, box offsets and objectness logits.
         *
         * Each level goes through a 1x1 stem, then two 3x3 towers: the class tower ends in a
         * 1x1 to the class count, the box tower in a 1x1 to four values and a 1x1 to one. The
         * node appends three values per level: "level<l>.classes", "level<l>.boxes" and
         * "level<l>.objectness".
         */
        class DecoupledHead final : public NN::Node
        {
            public:
                /**
                 * @brief Builds the branches over the per-sample shapes of the levels.
                 * @param levels One [H, W, C] shape per level, finest first.
                 * @param classes The number of class logits per position; at least one.
                 * @param hidden The width of the stems and towers; at least one.
                 * @return The node, or InvalidDescription for a bad level, classes or hidden.
                 */
                [[nodiscard]] static Result<std::unique_ptr<DecoupledHead>> create(
                    NN::ShapeSpan levels, int64_t classes, int64_t hidden,
                    NN::NormOptions norm) noexcept;

                /**
                 * @brief Traces every level and appends its class, box and objectness values.
                 * @return InvalidDescription unless @p inputs holds one value per level; @p outputs
                 *         is unchanged then.
                 */
                [[nodiscard]] Status forward(const NN::Trace& trace, NN::ValueSpan inputs,
                                             std::vector<Graph::ValueID>& outputs) override;

                /** @brief Walks the six groups of branches under @p prefix. */
                void walk_state(std::string_view prefix, const NN::StateVisitor& visitor) override;

                /** @brief Returns the number of levels the node was built over. */
                [[nodiscard]] std::size_t input_count() const noexcept override;

                /** @brief Lists three ports per level: classes, boxes, objectness. */
                [[nodiscard]] SushiRuntime::span<const NN::Port> ports() const noexcept override;

            private:
                /** @brief Binds the branches and the ports create() derived from them. */
                DecoupledHead(DecoupledBranches branches, std::vector<NN::Port> ports,
                              std::size_t level_count);

                /** @brief Holds the layers of every level. */
                DecoupledBranches branches_;

                /** @brief Holds the ports in the order forward() appends to them. */
                std::vector<NN::Port> ports_;

                /** @brief Holds how many values forward() reads. */
                std::size_t level_count_;
        };

        /**
         * @brief Builds a DecoupledHead node from "classes" and "hidden", with the optional
         *        "bn_epsilon" and "bn_momentum".
         * @param inputs One per-sample [H, W, C] shape per level, finest first.
         * @return The node, or InvalidDescription when @p inputs or a key does not fit.
         */
        [[nodiscard]] Result<std::unique_ptr<NN::Node>> make_decoupled_head_node(
            NN::ShapeSpan inputs, const Config::Value& attributes, std::string_view context) noexcept;

    } // namespace Vision::Blocks
} // namespace SushiAI
