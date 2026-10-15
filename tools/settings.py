# settings.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Holds the settings of one repository that the shared checkers read.

copy_shared_tools.py writes this file only when a repository has none, and never overwrites it.
"""

from __future__ import annotations

K_TIER_ORDER: tuple[str, ...] = ("core", "layers", "vision", "language")
K_CLOSED = False
K_MACRO_PREFIXES: tuple[str, ...] = ("SA_", "SUSHIAI_")
K_DOMAIN_TIERS: tuple[str, ...] = ("vision", "language")
K_SAME_TIER_EDGES: frozenset[tuple[str, str]] = frozenset({
    ("autograd", "core"),
    ("autograd", "graph"),
    ("config", "core"),
    ("darknet_reference", "config"),
    ("darknet_reference", "core"),
    ("data", "config"),
    ("data", "core"),
    ("eager", "autograd"),
    ("eager", "core"),
    ("eager", "graph"),
    ("engine_fixture", "core"),
    ("failure_expectations", "core"),
    ("graph", "core"),
    ("model_text", "config"),
    ("optim", "config"),
    ("optim", "core"),
    ("optim", "graph"),
    ("tensor_values", "core"),
    ("classifier_objective", "train"),
    ("io", "nn"),
    ("layer_rows", "nn"),
    ("layer_traces", "nn"),
    ("reference_files", "io"),
    ("sample_models", "nn"),
    ("train", "nn"),
    ("codecs", "imaging"),
    ("darknet_forward_cases", "detect"),
    ("darknet_forward_cases", "importers"),
    ("darknet_import_cases", "darknet_forward_cases"),
    ("darknet_import_cases", "detect"),
    ("darknet_import_cases", "importers"),
    ("darknet_training_cases", "darknet_forward_cases"),
    ("darknet_training_cases", "detect"),
    ("darknet_training_cases", "importers"),
    ("darknet_training_cases", "training"),
    ("datasets", "codecs"),
    ("datasets", "imaging"),
    ("datasets", "segment"),
    ("importers", "detect"),
    ("registration", "blocks"),
    ("registration", "datasets"),
    ("registration", "detect"),
    ("registration", "importers"),
    ("registration", "training"),
    ("segment", "imaging"),
    ("training", "detect"),
    ("training", "segment"),
    ("datasets", "text"),
    ("decoder_models", "registration"),
    ("generation", "text"),
    ("registration", "blocks"),
    ("registration", "datasets"),
    ("registration", "training"),
})
K_MODULE_FUNCTION = ""
K_PUBLIC_DEPENDS = "PUBLIC_DEPENDS"
K_PRIVATE_DEPENDS = "PRIVATE_DEPENDS"
K_INTERNAL_MARK = "INTERNAL"
