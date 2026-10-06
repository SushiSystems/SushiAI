/**************************************************************************/
/* SushiAI.h                                                              */
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
 * @file SushiAI.h
 * @brief Includes the full public interface of the SushiAI library.
 * @author Mustafa Garip
 */

#pragma once

#include "version.hpp"

#include "config/value.hpp"
#include "config/registry.hpp"
#include "config/model_config.hpp"
#include "config/train_config.hpp"

#include "core/common.hpp"
#include "core/ops.hpp"
#include "core/shape.hpp"
#include "core/allocation.hpp"

#include "graph/ir.hpp"
#include "graph/builder.hpp"
#include "graph/precision.hpp"
#include "graph/cast_insertion.hpp"
#include "graph/fusion.hpp"
#include "graph/memory_plan.hpp"
#include "graph/arena.hpp"
#include "graph/fusion_selector.hpp"
#include "graph/lowering.hpp"
#include "graph/step_recorder.hpp"
#include "graph/compiled_step.hpp"
#include "graph/rematerialise.hpp"

#include "autograd/backward.hpp"
#include "autograd/tape.hpp"
#include "autograd/rule.hpp"
#include "autograd/rules/all.hpp"
#include "autograd/rules/elementwise.hpp"
#include "autograd/rules/linalg.hpp"
#include "autograd/rules/activation.hpp"
#include "autograd/rules/loss.hpp"
#include "autograd/rules/shape.hpp"
#include "autograd/rules/spatial.hpp"
#include "autograd/rules/norm.hpp"
#include "autograd/gradcheck.hpp"

#include "nn/module.hpp"
#include "nn/layers.hpp"
#include "nn/ilayer.hpp"
#include "nn/layer_stack.hpp"
#include "nn/layer_registry.hpp"
#include "nn/loss.hpp"
#include "nn/init.hpp"

#include "optim/step_scalars.hpp"
#include "optim/loss_scaler.hpp"
#include "optim/optimizer.hpp"
#include "optim/sgd.hpp"
#include "optim/adamw.hpp"
#include "optim/optimizer_registry.hpp"

#include "data/dataset.hpp"

#include "io/checkpoint_format.hpp"
#include "io/checkpoint.hpp"

#include "train/objective.hpp"
#include "train/trainer.hpp"
