# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# data: the dataset interface, the synthetic datasets and their registry.

# The loader runs readers on threads of the standard library.
find_package(Threads REQUIRED)

sushiai_add_module(NAME data TIER core
    SOURCES
        source/dataset.cpp
        source/dataset_registry.cpp
        source/epoch.cpp
        source/epoch_permutation.cpp
        source/materialised_dataset.cpp
        source/threaded_loader.cpp
    PUBLIC_DEPENDS core config
    EXTERNAL_PUBLIC Threads::Threads)
