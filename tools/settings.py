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
K_SAME_TIER_EDGES: frozenset[tuple[str, str]] = frozenset()
K_MODULE_FUNCTION = "sushiai_add_module"
K_PUBLIC_DEPENDS = "PUBLIC_DEPENDS"
K_PRIVATE_DEPENDS = "PRIVATE_DEPENDS"
K_INTERNAL_MARK = "INTERNAL"
K_PRIVATE_DEPENDS_ALSO: tuple[str, ...] = ()
K_LOG_PREFIX = "SA"
K_CONSOLE_PATHS: tuple[str, ...] = (
    "applications/",
    "cli/sushiai/services/",
    "modules/vision/darknet_forward_cases/",
    "modules/vision/darknet_training_cases/",
)
K_DESIGN_TOPICS = True
K_FOLDER_CEILING = 60
K_CLOSED_GRACE_DAYS = 0
K_ARCHIVE_UNTRACKED = True
K_TOOLING_PACKAGE = 'sushicore'
K_TOOLING_MINIMUM_VERSION = '0.10.2'
K_TOOLING_SIBLING_SOURCE = 'sushicore/source'
K_LICENSE_HOLDER = 'Mustafa Garip & Sushi Systems'
K_LICENSE_COMPANY = 'Sushi Systems'
K_LICENSE_SITE = 'https://sushisystems.io'
K_LICENSE_OPEN_LINES = ('Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.', 'Commercial use requires a licence from Sushi Systems.')
K_LICENSE_CLOSED_LINES = ('All rights reserved. No licence is granted.',)
K_LICENSE_OPEN_BOX_LINES = ('Licensed under the PolyForm Noncommercial License 1.0.0 (the', '"License"); you may not use this file except in compliance with the', 'License. You may obtain a copy of the License at', '', '    https://polyformproject.org/licenses/noncommercial/1.0.0', '', 'Noncommercial use is free. Commercial use requires a separate licence', 'from Sushi Systems; see COMMERCIAL.md. The software is provided', '"as is", without warranty of any kind.')
K_LICENSE_CLOSED_BOX_LINES = ('All rights reserved.', '', 'This file is proprietary and confidential. No licence is granted.', 'Copying, modifying, distributing or using it, in whole or in part,', 'requires prior written permission from Sushi Systems.')
K_LOGGER_MACRO_PREFIX = 'SL'
K_LOGGER_NAMESPACE = 'SushiLog'
K_LOGGER_C_PREFIX = 'sl_'
K_MODULE_MANIFEST = 'sushi-module.toml'
K_MODULE_DEPENDENCY_FILE = 'cli/sushistack.deps.toml'
K_MODULE_HEADINGS = ('## Standalone', '## With SushiHub')
K_MODULE_WORKSPACE_READ = '\\bhub (?:install|add|link|doctor|status|pin)\\b|SUSHISTACK_HOME|\\.sushistack\\b|workspace\\.toml'
