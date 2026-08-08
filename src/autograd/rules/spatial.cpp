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
    }
}
