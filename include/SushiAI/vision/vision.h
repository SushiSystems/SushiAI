/**************************************************************************/
/* vision.h                                                               */
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
 * @file vision.h
 * @brief Includes SushiAI and every public header of the vision domain.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/SushiAI.h>

#include "imaging/affine.hpp"
#include "imaging/boxes.hpp"
#include "imaging/dataset_partners.hpp"
#include "imaging/flip.hpp"
#include "imaging/gray.hpp"
#include "imaging/hsv.hpp"
#include "imaging/image.hpp"
#include "imaging/label_set.hpp"
#include "imaging/labels.hpp"
#include "imaging/letterbox.hpp"
#include "imaging/mosaic.hpp"
#include "imaging/partner_source.hpp"
#include "imaging/pipeline.hpp"
#include "imaging/polygon.hpp"
#include "imaging/random.hpp"
#include "imaging/raw_cache.hpp"
#include "imaging/recent_partners.hpp"
#include "imaging/rectangle.hpp"
#include "imaging/resize.hpp"

#include "detect/anchor_leaves.hpp"
#include "detect/anchors.hpp"
#include "detect/assigner.hpp"
#include "detect/assigner_config.hpp"
#include "detect/assignment.hpp"
#include "detect/average_precision.hpp"
#include "detect/box_coding.hpp"
#include "detect/box_format.hpp"
#include "detect/detection_row.hpp"
#include "detect/detections_nms.hpp"
#include "detect/detections_topk.hpp"
#include "detect/ground_truth.hpp"
#include "detect/iou.hpp"
#include "detect/level_rows.hpp"
#include "detect/prediction_shape.hpp"
#include "detect/target_leaves.hpp"
#include "detect/top_k.hpp"

#include "blocks/bottleneck.hpp"
#include "blocks/c3k.hpp"
#include "blocks/c3k2.hpp"
#include "blocks/conv_norm.hpp"
#include "blocks/sppf.hpp"

#include "datasets/mnist.hpp"

#include "registration.hpp"
