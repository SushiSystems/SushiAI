/**************************************************************************/
/* spatial.cpp                                                           */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                https://github.com/SushiSystems/SushiAI                 */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/rules/spatial.hpp>

// IM2COL and COL2IM are each other's adjoint (CV-1): a convolution is IM2COL
// followed by MATMUL, matmul_backward already produces the weight gradient,
// and col2im *is* im2col's transpose - which is why this is the only rule
// convolution needs. Tasks 12-13 extend this table with pooling and the
// channel-structural ops.

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

            // col2im *is* im2col's adjoint: a pixel read by several windows
            // gets the sum of their gradients, which is what makes this one
            // call rather than a scatter loop. SushiBLAS writes it as a
            // gather, so the sum's order is fixed and the result is
            // bit-reproducible (CV-6).
            //
            // col2im_with_extents, not the derived-extent col2im, and the
            // extents are read out of the forward node's slots 8 and 9 rather
            // than recomputed. Recomputing them means assuming the padding was
            // symmetric, and an asymmetric window grid can derive to its own
            // transpose with the same window count - which every shape check
            // downstream accepts, so the gradient would come back wrong rather
            // than throwing. The forward wrote its real grid into the node;
            // this reads it.
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

            // The adjoint of the adjoint, over the same window grid the
            // forward used - read from slots 8 and 9 for the same reason
            // im2col_backward reads them: the column matrix this node consumed
            // may have come from an asymmetrically padded im2col, whose grid
            // the geometry cannot reproduce and can silently transpose.
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

            // The indices are this node's second *result*, so the rule reads
            // exactly what its own forward wrote - there is no way to pair a
            // gradient with another pool's argmax.
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

            // No index and no saved forward: every element of a window gets
            // the same share, so the divisor is the window area alone. This is
            // why it is a separate function from the max rule rather than the
            // same one with a flag.
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

            // One slice per operand, at that operand's running offset. Walked
            // in operand order rather than looked up anywhere, because the
            // offset is defined only by the order the forward concatenated in.
            const Graph::Node& node = tape.node();
            int64_t offset = 0;
            for (std::size_t i = 0; i < node.inputs.size(); ++i)
            {
                const Shape& part = tape.graph().value(node.inputs[i]).shape;

                // The forward emitter already refuses anything but rank 4, so
                // this holds today. It is checked anyway because it is the one
                // rule that indexes a raw dimension, and reading shape[3] off
                // a rank-2 operand would not throw - it would read whatever
                // Shape keeps past its rank and slice the gradient by it.
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

            // The adjoint of a slice is a scatter into zeros. Expressed as a
            // concat of up to three parts rather than as a new identity: the
            // ranges before and after the slice are zeros, and CONCAT_CHANNELS
            // already knows how to place a run of channels at an offset.
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
