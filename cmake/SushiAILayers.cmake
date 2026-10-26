# SushiAILayers.cmake
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The tier table: which tier every module belongs to, the order of the tiers,
# and the library each tier is archived into. cmake/SushiAIModule.cmake reads
# it; tools/layering/check_layering.py holds the same table for the includes.
# The design is docs/design/DOMAIN_MODULES/README.md.

# Lowest first. A module may depend on a module of its own tier or of a lower one.
set(SUSHIAI_TIER_ORDER core layers vision language)

# Tiers that are domains. Every domain ranks directly above `layers`, and no
# module of one domain may depend on a module of another.
set(SUSHIAI_DOMAIN_TIERS vision language)

# Pairs: tier, then the library target its modules are archived into.
set(SUSHIAI_TIER_LIBRARY
    core    sushiai
    layers  sushiai
    vision  sushiai_vision
    language  sushiai_language
)

# Pairs: module key, then its tier. A module of a domain tier is keyed
# <tier>/<module>, so two domains may each have a `blocks` or a `datasets`.
set(SUSHIAI_MODULE_TIERS
    log                  core
    profile              core
    profile_records      core
    failure              core
    dtype                core
    hash                 core
    shape                core
    ops                  core
    seam                 core
    execution            core
    allocation           core
    version              core
    graph                core
    config               core
    autograd             core
    optim                core
    data                 core
    eager                core
    distribution         core
    engine_fixture       core
    failure_expectations core
    tensor_values        core
    seeded_noise         core
    scratch_files        core
    model_text           core
    darknet_reference    core
    nn                   layers
    io                   layers
    train                layers
    umbrella             layers
    layer_rows           layers
    layer_traces         layers
    sample_models        layers
    reference_files      layers
    classifier_objective layers
    vision/imaging       vision
    vision/codecs        vision
    vision/detect        vision
    vision/segment       vision
    vision/blocks        vision
    vision/datasets      vision
    vision/importers     vision
    vision/training      vision
    vision/registration  vision
    vision/umbrella      vision
    vision/darknet_forward_cases vision
    vision/darknet_import_cases vision
    vision/darknet_training_cases vision
    language/text        language
    language/blocks      language
    language/datasets    language
    language/training    language
    language/generation  language
    language/registration language
    language/umbrella    language
    language/decoder_models language
)
