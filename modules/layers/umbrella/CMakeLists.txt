# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# umbrella: the header that includes every public header of core and layers.

# SushiAI.hpp includes the distribution headers under SA_DISTRIBUTED, which the module defines;
# a build without SA_ENABLE_DISTRIBUTED names distribution unbuilt, so the umbrella skips it.
sushiai_add_module(NAME umbrella TIER layers
    PUBLIC_DEPENDS
        allocation
        autograd
        config
        data
        distribution
        dtype
        eager
        hash
        execution
        failure
        graph
        io
        log
        nn
        ops
        optim
        seam
        shape
        train
        version)
