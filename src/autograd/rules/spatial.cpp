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
            // col2im_with_extents, not the derived-extent col2im: the derived
            // overload assumes symmetric padding, but the forward's real
            // output extents are already sitting in the column matrix's own
            // shape, so re-deriving them would silently misbehave on an
            // asymmetrically-padded layer.
            const Graph::Node& node = tape.node();
            const Graph::ValueId x = node.inputs[0];
            const Graph::ConvGeometry g = geometry_of(node);
            const Shape image_shape = tape.graph().value(x).shape;
            const Shape cols_shape = tape.graph().value(node.outputs[0]).shape;

            const int64_t windows_per_image = cols_shape[0] / image_shape[0];
            const int64_t out_w = Graph::output_extent(image_shape[2], g.pad_left, g.pad_left,
                                                        g.dilation_w, g.kernel_w, g.stride_w);
            const int64_t out_h = windows_per_image / out_w;

            tape.contribute_input(0, tape.builder().col2im_with_extents(
                dy, image_shape, g, out_h, out_w));
        }

        void col2im_backward(const GradientTape& tape)
        {
            if (!tape.input_requires_grad(0)) return;

            const Graph::ValueId dy = tape.output_grad(0);
            if (dy == Graph::NO_VALUE) return;

            // The adjoint of the adjoint. The extents come from the column
            // matrix this node consumed rather than from the geometry, because
            // that operand may have been produced by an asymmetrically padded
            // im2col whose extents the geometry cannot reproduce.
            const Graph::Node& node = tape.node();
            const Graph::ConvGeometry g = geometry_of(node);
            const Shape image = tape.graph().value(node.outputs[0]).shape;
            const int64_t windows = tape.graph().value(node.inputs[0]).shape[0];
            const int64_t per_image = windows / image[0];
            const int64_t out_w = Graph::output_extent(image[2], g.pad_left, g.pad_left,
                                                       g.dilation_w, g.kernel_w, g.stride_w);

            tape.contribute_input(0, tape.builder().im2col_with_extents(
                dy, g, per_image / out_w, out_w));
        }
    }
}
