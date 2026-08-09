/**************************************************************************/
/* rematerialise.hpp                                                      */
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

#pragma once

/**
 * @file rematerialise.hpp
 * @brief Recomputing a convolution's column matrix instead of storing it (CV-10).
 *
 * A pure IR-to-IR function in the host lane, the same shape as
 * @c graph/cast_insertion.hpp and @c graph/fusion.hpp.
 *
 * @section why The number this exists for
 *
 * @c Ops::IM2COL expands an @c [N,H,W,C] image into an
 * @c [N*H_o*W_o, kernel_h*kernel_w*C] matrix, so its result is larger than its
 * operand by the window area - a factor of nine for the 3x3 convolutions a
 * ResNet is mostly made of. ResNet-18's first block at batch 128 holds
 * `128*32*32*64*9*4` bytes, about 302 MB, in that one buffer, and the network
 * has twenty convolutions. The backward pass reads the column matrix again to
 * form the weight gradient, so every one of those buffers is live from its
 * forward GEMM all the way to its backward GEMM: twenty at once.
 *
 * Recomputing it beside the backward GEMM costs one extra gather - the cheap
 * half of a convolution, next to the GEMM it feeds - and drops the stored
 * copy's live range to the gap between the forward gather and the forward
 * GEMM, which are adjacent. One column matrix is live at a time instead of
 * twenty.
 *
 * @section not_general Why there is no general rematerialisation pass here
 *
 * Deliberately named for the one operation it serves. A general
 * "rematerialise what is cheap to recompute" pass would need a cost model
 * weighing recomputation against bytes saved, would have to be told how many
 * bytes are actually available to decide anything, and would have exactly this
 * one client. @c IM2COL is exempt from needing that model because the
 * arithmetic above is not close: its buffer is a whole window area larger than
 * the value it was gathered from, and the recomputation is a gather rather
 * than a GEMM. Nothing else in the IR has that ratio, so nothing else would
 * benefit from generalising it.
 *
 * @section position Where it runs
 *
 * After @c Autograd::differentiate() - there is no second consumer to
 * rematerialise for until the backward pass exists - and before
 * @c MemoryPlan::build(), which is the pass that turns the shortened live
 * range into fewer bytes. Running it after planning would change nothing.
 *
 * @section equivalence What it does not change
 *
 * The clone reads the same operands and carries the same @c Attributes as the
 * node it copies, so it computes the same numbers. The attributes are copied
 * wholesale rather than rebuilt from the geometry, and that is load-bearing:
 * @c IM2COL carries its output extents in slots 8 and 9 precisely because they
 * cannot be re-derived (see @c Attributes::CAPACITY), and a clone that
 * re-derived them under an asymmetric padding would gather a transposed window
 * grid with the same element count - every shape check would agree and the
 * numbers would be wrong.
 */

#include <SushiAI/graph/ir.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /**
         * @brief Recomputes every stored @c IM2COL result at its later consumers.
         *
         * For each @c IM2COL whose result has more than one consumer, inserts
         * one clone of the gather immediately before the second consumer,
         * rewires every consumer from the second onwards to read the clone, and
         * demotes the original to @c Lifetime::SCRATCH - it now dies at its
         * first reader.
         *
         * One clone shared by the later consumers rather than one each: they
         * are the backward pass's readers of a single forward value and run
         * adjacently, so a clone apiece would buy no liveness and cost a gather
         * apiece.
         *
         * An @c IM2COL with one consumer or none is left exactly as it was.
         * That is the inference case: nothing was differentiated, there is no
         * second reader, and cloning would be a free slowdown.
         *
         * Value ids are preserved, so any id a caller holds goes on naming the
         * same value. Node ids are not, because a recomputation has to be
         * inserted ahead of its consumers rather than appended - see
         * @c Graph::insert_node().
         *
         * @param graph The graph to rewrite, in place.
         * @throws Error If the graph is not in topological insertion order.
         */
        void rematerialise_im2col(Graph& graph);

    } // namespace Graph
} // namespace SushiAI
