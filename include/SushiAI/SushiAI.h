/**************************************************************************/
/* SushiAI.h                                                              */
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
 * @file SushiAI.h
 * @brief Master header for the SushiAI library.
 *
 * Includes every public header, so a consumer writes one include and gets the
 * whole library. The ordering below is the dependency order of the layers:
 * vocabulary, the graph IR that autograd and the later passes transform, the
 * lowering that turns that IR into SushiBLAS work, then the layers, losses,
 * optimizers, data and training loop built on top.
 *
 * Two lanes live under this roof, and the split is worth knowing about:
 *
 * - The @b host lane - @c core/common.hpp, @c core/ops.hpp,
 *   @c core/shape.hpp, @c graph/ir.hpp, @c graph/builder.hpp,
 *   @c graph/memory_plan.hpp, @c autograd/, @c nn/module.hpp,
 *   @c nn/layers.hpp, @c nn/loss.hpp - pulls in no SushiBLAS and no SYCL.
 *   Shape inference, graph topology, the differentiation transform, the memory
 *   planner and the parameter walk are all testable without a device, and this
 *   repository's unit tests include those headers directly for exactly that
 *   reason.
 * - The @b device lane - everything else, including @c core/allocation.hpp -
 *   reaches SushiBLAS and therefore SYCL.
 *
 * Including this master header gets both. Include a host-lane header directly
 * when you want to stay off the device compiler.
 */

#include "version.hpp"

#include "core/common.hpp"
#include "core/ops.hpp"
#include "core/shape.hpp"
#include "core/allocation.hpp"

#include "graph/ir.hpp"
#include "graph/builder.hpp"
#include "graph/memory_plan.hpp"
#include "graph/arena.hpp"
#include "graph/lowering.hpp"
#include "graph/step_recorder.hpp"
#include "graph/compiled_step.hpp"

#include "autograd/backward.hpp"
#include "autograd/gradcheck.hpp"

#include "nn/module.hpp"
#include "nn/layers.hpp"
#include "nn/loss.hpp"
#include "nn/init.hpp"

#include "optim/step_scalars.hpp"
#include "optim/optimizer.hpp"
#include "optim/sgd.hpp"
#include "optim/adamw.hpp"

#include "data/dataset.hpp"

#include "train/trainer.hpp"
