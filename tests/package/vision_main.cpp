/**************************************************************************/
/* vision_main.cpp                                                        */
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
 * @file vision_main.cpp
 * @brief Registers the vision domain through an installed SushiAI package, out of tree.
 * @author Mustafa Garip
 */

#include <cstdio>
#include <SushiAI/vision/vision.h>

/** @brief Shortens the SushiAI namespace. */
namespace sa = SushiAI;

/**
 * @brief Registers the layers and the vision rows, then checks that "c3k2" and "mnist" resolve.
 * @return 0 when both names resolve, 1 otherwise.
 */
int main()
{
    sa::NN::NodeRegistry nodes;
    sa::Data::DatasetRegistry datasets;

    sa::Status status = sa::NN::register_layers(nodes);
    if (status.is_ok())
        status = sa::Vision::register_nodes(nodes);
    if (status.is_ok())
        status = sa::Vision::register_datasets(datasets);
    if (!status.is_ok())
    {
        std::fprintf(stderr, "FAIL: registration returned %s\n",
                     sa::status_code_name(status.code()));
        return 1;
    }

    if (nodes.find_node("c3k2") == nullptr || datasets.find("mnist") == nullptr)
    {
        std::fprintf(stderr, "FAIL: c3k2 or mnist is not registered\n");
        return 1;
    }

    std::printf("OK: SushiAI %s registers the vision blocks and MNIST through a package\n",
                sa::version_string());
    return 0;
}
