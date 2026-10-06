/**************************************************************************/
/* spatial.cpp                                                            */
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
 * @file spatial.cpp
 * @brief Implements backward differentiation rules for spatial and pooling operations.
 * @author Mustafa Garip
 */

#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/rules/spatial.hpp>

namespace SushiAI
{
    namespace Autograd
    {
        namespace
        {
            /** @brief Reads CV-5's eight slots back out of a node's attributes. */
            Graph::ConvGeometry geometry_of(const Graph::Node& node) noexcept
            {
                Graph::ConvGeometry g;
                g.kernel_h   = static_cast<int64_t>(node.attributes.get(0));
                g.kernel_w   = static_cast<int64_t>(node.attributes.get(1));
                g.stride_h   = static_cast<int64_t>(node.attributes.get(2));
                g.stride_w   = static_cast<int64_t>(node.attributes.get(3));
                g.dilation_h = static_cast<int64_t>(node.attributes.get(4));
                g.dilation_w = static_cast<int64_t>(node.attributes.get(5));
                g.pad_top    = static_cast<int64_t>(node.attributes.get(6));
                g.pad_left   = static_cast<int64_t>(node.attributes.get(7));
                return g;
            }

            /** @brief Reads the forward's own window grid out of slots 8-9. */
            void extents_of(const Graph::Node& node, const char* what,
                            int64_t& out_h, int64_t& out_w)
            {
                out_h = static_cast<int64_t>(node.attributes.get(8));
                out_w = static_cast<int64_t>(node.attributes.get(9));

                SA_THROW_IF(out_h <= 0 || out_w <= 0,
                            "{}'s gradient needs the forward's window grid, but the node "
                            "carries {}x{} in attribute slots 8 and 9",
                            what, out_h, out_w);
            }
        }

        void im2col_backward(const GradientTape& tape)
        {
            if (!tape.input_requires_grad(0)) return;

            const Graph::ValueId dy = tape.output_grad(0);
            if (dy == Graph::NO_VALUE) return;

            // Col2im adjoint accumulates window gradients using recorded forward window extents.
            const Graph::Node& node = tape.node();
            const Graph::ValueId x = node.inputs[0];
            const Graph::ConvGeometry g = geometry_of(node);
            const Shape image_shape = tape.graph().value(x).shape;

            int64_t out_h = 0;
            int64_t out_w = 0;
            extents_of(node, "IM2COL", out_h, out_w);

            tape.contribute_input(0, tape.builder().col2im_with_extents(
                dy, image_shape, g, out_h, out_w));
        }

        void col2im_backward(const GradientTape& tape)
        {
            if (!tape.input_requires_grad(0)) return;

            const Graph::ValueId dy = tape.output_grad(0);
            if (dy == Graph::NO_VALUE) return;

            // Reconstruct im2col adjoint using recorded forward window grid extents.
            const Graph::Node& node = tape.node();
            const Graph::ConvGeometry g = geometry_of(node);

            int64_t out_h = 0;
            int64_t out_w = 0;
            extents_of(node, "COL2IM", out_h, out_w);

            tape.contribute_input(0, tape.builder().im2col_with_extents(dy, g, out_h, out_w));
        }

        void maxpool2d_backward(const GradientTape& tape)
        {
            if (!tape.input_requires_grad(0)) return;

            const Graph::ValueId dy = tape.output_grad(0);
            if (dy == Graph::NO_VALUE) return;

            // Maxpool backward routes gradients through argmax indices from the forward node output.
            const Graph::Node& node = tape.node();
            tape.contribute_input(0, tape.builder().maxpool2d_backward(
                dy, node.outputs[1], tape.graph().value(node.inputs[0]).shape,
                geometry_of(node)));
        }

        void avgpool2d_backward(const GradientTape& tape)
        {
            if (!tape.input_requires_grad(0)) return;

            const Graph::ValueId dy = tape.output_grad(0);
            if (dy == Graph::NO_VALUE) return;

            // Avgpool backward distributes incoming gradients uniformly across the window area.
            const Graph::Node& node = tape.node();
            tape.contribute_input(0, tape.builder().avgpool2d_backward(
                dy, tape.graph().value(node.inputs[0]).shape, geometry_of(node)));
        }

        void upsample2d_backward(const GradientTape& tape)
        {
            if (!tape.input_requires_grad(0)) return;

            const Graph::ValueId dy = tape.output_grad(0);
            if (dy == Graph::NO_VALUE) return;

            const Graph::Node& node = tape.node();
            tape.contribute_input(0, tape.builder().upsample2d_backward(
                dy, tape.graph().value(node.inputs[0]).shape,
                static_cast<int64_t>(node.attributes.get(0)),
                static_cast<int64_t>(node.attributes.get(1))));
        }

        void spatial_mean_backward(const GradientTape& tape)
        {
            if (!tape.input_requires_grad(0)) return;

            const Graph::ValueId dy = tape.output_grad(0);
            if (dy == Graph::NO_VALUE) return;

            const Graph::Node& node = tape.node();
            tape.contribute_input(0, tape.builder().spatial_mean_backward(
                dy, tape.graph().value(node.inputs[0]).shape));
        }

        void concat_channels_backward(const GradientTape& tape)
        {
            const Graph::ValueId dy = tape.output_grad(0);
            if (dy == Graph::NO_VALUE) return;

            // Channel concatenation backward extracts operand slices along cumulative channel offsets.
            const Graph::Node& node = tape.node();
            int64_t offset = 0;
            for (std::size_t i = 0; i < node.inputs.size(); ++i)
            {
                const Shape& part = tape.graph().value(node.inputs[i]).shape;

                // Enforce rank-4 NHWC shape before indexing channel dimension.
                SA_THROW_IF(part.rank() != 4,
                            "concat_channels_backward reads the channel extent of a rank-4 "
                            "[N,H,W,C] operand; operand {} is {}", i, part.to_string());

                const int64_t channels = part[3];
                if (tape.input_requires_grad(i))
                    tape.contribute_input(i, tape.builder().slice_channels(dy, offset, channels));
                offset += channels;
            }
        }

        void slice_channels_backward(const GradientTape& tape)
        {
            if (!tape.input_requires_grad(0)) return;

            const Graph::ValueId dy = tape.output_grad(0);
            if (dy == Graph::NO_VALUE) return;

            // Slice adjoint pads zeros around the active channel range via channel concatenation.
            const Graph::Node& node = tape.node();
            const Shape in = tape.graph().value(node.inputs[0]).shape;
            const int64_t offset = static_cast<int64_t>(node.attributes.get(0));
            const int64_t taken = tape.graph().value(node.outputs[0]).shape[3];
            const DType dtype = tape.graph().value(dy).dtype;

            std::vector<Graph::ValueId> parts;
            if (offset > 0)
                parts.push_back(tape.builder().zeros_like(
                    dy, Shape({in[0], in[1], in[2], offset}), dtype));
            parts.push_back(dy);
            if (offset + taken < in[3])
                parts.push_back(tape.builder().zeros_like(
                    dy, Shape({in[0], in[1], in[2], in[3] - offset - taken}), dtype));

            tape.contribute_input(0, parts.size() == 1
                ? dy
                : tape.builder().concat_channels(
                      SushiRuntime::span<const Graph::ValueId>(parts.data(), parts.size())));
        }
    }
}
