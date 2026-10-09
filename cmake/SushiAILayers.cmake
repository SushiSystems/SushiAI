# SushiAILayers.cmake
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The tier table: which tier every module belongs to, the order of the tiers,
# and the library each tier is archived into. cmake/SushiAIModule.cmake reads
# it; tools/layering/check_layering.py holds the same table for the includes.
# The design is docs/design/DOMAIN_MODULES.md.

# Lowest first. A module may depend on a module of its own tier or of a lower one.
set(SUSHIAI_TIER_ORDER core layers vision)

# Tiers that are domains. Every domain ranks directly above `layers`, and no
# module of one domain may depend on a module of another.
set(SUSHIAI_DOMAIN_TIERS vision)

# Pairs: tier, then the library target its modules are archived into.
set(SUSHIAI_TIER_LIBRARY
    core    sushiai
    layers  sushiai
    vision  sushiai_vision
)

# Pairs: module key, then its tier. A module of a domain tier is keyed
# <tier>/<module>, so two domains may each have a `blocks` or a `datasets`.
set(SUSHIAI_MODULE_TIERS
    core                 core
    version              core
    graph                core
    config               core
    autograd             core
    optim                core
    data                 core
    eager                core
    nn                   layers
    io                   layers
    train                layers
    umbrella             layers
    vision/imaging       vision
    vision/detect        vision
    vision/blocks        vision
    vision/datasets      vision
    vision/registration  vision
    vision/umbrella      vision
)
