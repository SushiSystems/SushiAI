# Estate audit: sushiai

**Status:** Complete. Read-only audit of 2026-10-05; nothing in the repository was changed.

One auditing agent per dimension read the tree and a second agent checked each finding against
the files. A finding marked *confirmed* was re-read by the reviewer; *unclear* means the reviewer
could not settle it; *found by review* means the first pass missed it and nobody re-checked it.
Refuted findings are listed and not counted. No build, test or project CLI was run.

| Dimension | High | Medium | Low | Refuted |
| --- | --- | --- | --- | --- |
| Licence | 1 | 10 | 6 | 0 |
| Command line | 2 | 11 | 9 | 0 |
| Documentation | 4 | 14 | 3 | 0 |
| Layout and hygiene | 3 | 12 | 8 | 0 |
| Code shape | 4 | 15 | 3 | 0 |

## Licence

### Facts

- Tracked files: 232 (85 .cpp, 55 .hpp, 1 .h, 24 .py, 34 .md, 9 .cmake, 6 CMakeLists.txt, 4 .toml, 5 .json, 2 .in, 1 .yml, Doxyfile, LICENSE). Remote: https://github.com/SushiSystems/SushiAI.git. No local git tags.
- D:/Projects/sushiai/LICENSE is the Apache License 2.0 full text, 190 lines; its appendix line 178 reads 'Copyright 2025-present Mustafa Garip & Sushi Systems'.
- No other licence document is tracked: no NOTICE, COPYING, docs/LICENSE, cli/LICENSE or third-party notice file. docs/LICENSE existed once and was removed on 2026-09-16 (docs/agent/2026_09_16_DOCS_AND_ROOT_HYGIENE/REPORT.md:35).
- Licence history of LICENSE: CC0 1.0 Universal at the initial commit d45522b (2025-06-08); changed in f06c42a (2025-07-02, 'Update LICENSE'; content not read); MIT before 7988fd6 (2026-08-01, 'chore(license): relicense from MIT to Apache-2.0'), recorded as decision D7 in docs/design/ROADMAP.md:176-181.
- Header variant A (full boxed Apache-2.0 notice, 'Copyright (c) 2026-present Mustafa Garip & Sushi Systems', project SushiAI, URLs github.com/SushiSystems/SushiAI and sushisystems.io): 106 files = 50 .cpp + 55 .hpp + 1 .h. Examples: include/SushiAI/SushiAI.h, src/config/value.cpp, and all of include/SushiAI/**/*.hpp.
- Header variant B (two '#' lines: 'Copyright (c) 2026-present Mustafa Garip & Sushi Systems' / 'Licensed under the Apache License, Version 2.0. See LICENSE.'): 6 .py files, all under tools/. Examples: tools/common/checker.py, tools/documentation/check_source_comments.py, tools/layering/check_layering.py.
- Header variant C (truncated box: file name, 'This file is part of: SushiAI', no URL, no copyright, no licence text): 33 .cpp files under tests/functional/. Examples: tests/functional/unit/test_ops.cpp, tests/functional/unit/test_ir.cpp, tests/functional/integration/test_training.cpp.
- No header at all, C++: 2 files, tests/functional/integration/test_engine_smoke.cpp and tests/functional/unit/test_version.cpp. So 35 of 85 .cpp carry no licence or copyright statement; 0 of 56 headers (.hpp/.h) lack one.
- No header at all, Python: 18 of 24 .py files, namely all 10 under cli/sushiai/ and all 8 under cli/tests/. Examples: cli/sushiai/cli.py, cli/sushiai/services/project.py, cli/tests/conftest.py.
- No header at all, build and config: 9 cmake/*.cmake, 6 CMakeLists.txt, cmake/SushiAIConfig.cmake.in, tests/fixtures/tiny_train.json.in, 4 .toml (sushi-module.toml, cli/pyproject.toml, cli/config.toml, cli/sushistack.deps.toml), .github/workflows/ci.yml, Doxyfile.
- Machine-readable declarations: cli/pyproject.toml has no `license` field and no classifiers (name sushiai-cli, version 0.1.0, author 'Mustafa Garip <mustafagarip@sushisystems.io>'). sushi-module.toml has no licence key. Doxyfile has none (PROJECT_NUMBER 'v1 (unreleased)'). CMakeLists.txt:16 is project(SushiAI VERSION 0.1.0) with no licence metadata and no install rule for LICENSE. No package.json, Dockerfile, conda file, vcpkg.json, setup.py or requirements file is tracked.
- Prose statements of the licence: README.md:575-577 ('licensed under the Apache License, Version 2.0'); docs/CONTRIBUTING.md:12-13 (contributions are licensed under Apache-2.0) and :103-128 (the canonical Apache header template); docs/design/ROADMAP.md:171-181, 196, 378-380, 672 (D5 portBLAS attribution, D7 relicence).
- Apache wording also appears in agent and archive documents: docs/agent/2026_09_16_CODE_STYLE_AND_COMMENTS/PLAN.md:16, REPORT.md:13, SPEC.md:22-33; docs/archive/superpowers/plans/2026-08-07-config-driven-training-plan.md:35.
- Copyright holder is the same string everywhere, 'Mustafa Garip & Sushi Systems'. Years differ: '2025-present' in LICENSE:178, '2026-present' in all 112 file headers and in the docs/CONTRIBUTING.md:116 template.
- @author: 93 occurrences in source, all 'Mustafa Garip'.
- git shortlog -sne HEAD: 117 commits 'Mustafa Garip <mustafa.garip3434@hotmail.com>', 9 commits 'Mustafa Garip <51916290+sushimg@users.noreply.github.com>'. One person under two addresses; no other human author. Committers: the same two plus 'GitHub <noreply@github.com>' on 7 web commits. First commit 2025-06-08, last 2026-10-04.
- Co-Authored-By trailers in commit bodies: Claude Opus 5 (1M context) 54, Claude Opus 5 39, Claude Sonnet 5 6, Claude Fable 5 1, Claude Opus 5.5 1, all <noreply@anthropic.com>. 101 trailers in total.
- No vendored third-party code: there is no third_party/ or dependencies/ directory in the tree, and no FetchContent or ExternalProject call in any tracked CMake file.
- C++ dependencies resolved at configure time: nlohmann_json (MIT; find_package at src/CMakeLists.txt:111, included only by src/config/value.cpp:36, declared in cli/sushistack.deps.toml:33-37 as apt nlohmann-json3-dev / vcpkg nlohmann-json); GoogleTest (BSD-3-Clause; tests/CMakeLists.txt:28, tests only). Licences stated from knowledge of the upstream projects, not read from an installed copy.
- Sibling Sushi modules: SushiRuntime (cmake/Runtime.cmake:27) and SushiBLAS (cmake/BLAS.cmake:17), declared in cli/sushistack.deps.toml:31. Both are Apache-2.0 today per docs/design/ROADMAP.md:181. hwloc reaches SushiAI only transitively through SushiRuntime (cmake/Vcpkg.cmake:3, examples/CMakeLists.txt:16-19).
- Python dependencies (cli/pyproject.toml:11-16): typer>=0.12 (MIT), rich>=13.0 (MIT), tomli>=2.0 on Python < 3.11 (MIT), sushicore>=0.7.0 (Sushi Systems, taken from PyPI per docs/reference/CHANGELOG.md:11), setuptools>=61.0 as build backend (MIT). CI also installs pytest and pyflakes (.github/workflows/ci.yml:178). Ranges, not exact pins.
- The SYCL toolchain (AdaptiveCpp or oneAPI, cmake/SushiAISyclTarget.cmake:19) and vcpkg are provisioned by the SushiStack workspace, not by this repository; nothing of theirs is tracked here.
- Release state: docs/reference/CHANGELOG.md has only an 'Unreleased' section; no tag exists locally; cli/pyproject.toml and CMakeLists.txt both say 0.1.0. No evidence in the tree that sushiai-cli was uploaded to PyPI.

### Corrections from review

- F4 recommendation: 'all 147 C++ and Python source files' is wrong. 147 = 106 + 6 + 33 + 2 and leaves out the 18 headerless Python files; the tracked total is 141 C++ (85 .cpp, 55 .hpp, 1 .h) + 24 .py = 165.
- Licence history fact: 'f06c42a (content not read); MIT before 7988fd6' leaves the MIT start date open. git show f06c42a:LICENSE opens 'MIT License / Copyright (c) 2025-present Mustafa Garip & Sushi Systems', so MIT ran from 2025-07-02 to 2026-08-01.
- '@author: 93 occurrences in source, all Mustafa Garip' is 92. The 93rd match is a string literal inside tools/documentation/check_source_comments.py, not an author tag.
- Tracked-file breakdown lists 228 of the 232 files; it omits .gitignore, .editorconfig, .clang-format and cli/sushiai/py.typed. The total of 232 is right.
- F6 cites cli/pyproject.toml:8 for the author address; the authors line is :9 (line 8 is description).
- 'Header variant A ... all of include/SushiAI/**/*.hpp' as the home of the 55 .hpp files is loose: include/ holds 52 C++ files (51 .hpp + SushiAI.h); the other 4 .hpp sit under tests/common/. The count of 55 is right.

### Findings

#### L1. [high] The tree has been public under CC0, then MIT, then Apache-2.0, and none of those grants can be withdrawn

Evidence: git log -- LICENSE: d45522b 2025-06-08 'Initial commit' (git show d45522b:LICENSE opens 'Creative Commons Legal Code / CC0 1.0 Universal'), f06c42a 2025-07-02 'Update LICENSE', 7988fd6 2026-08-01 'chore(license): relicense from MIT to Apache-2.0'. docs/design/ROADMAP.md:176 records D7. Remote is github.com/SushiSystems/SushiAI.

Recommendation: Treat the new licence as applying from the relicensing commit forward. State in the design document that every commit up to that point stays available under the licence it carried (Apache-2.0 back to 7988fd6, MIT and CC0 before). Confirm with the owner whether the GitHub repository is public and whether forks exist; that decides how much of the current code is already out under Apache-2.0.

Review: confirmed. git log -- LICENSE shows d45522b (CC0), f06c42a, 7988fd6; ROADMAP.md:176 records D7. Correction to the audit's own gap: f06c42a already holds the MIT text ('Copyright (c) 2025-present Mustafa Garip & Sushi Systems'), so the sequence is CC0 2025-06-08, MIT 2025-07-02, Apache-2.0 2026-08-01.

#### L2. [medium] 35 C++ test files carry no copyright or licence notice

Evidence: 33 files under tests/functional/ open with a six-line box that stops after 'This file is part of: SushiAI' (e.g. tests/functional/unit/test_ops.cpp:1-6, tests/functional/unit/test_ir.cpp, tests/functional/integration/test_training.cpp). 2 files have no box: tests/functional/integration/test_engine_smoke.cpp, tests/functional/unit/test_version.cpp. The source-comments skill and docs/CONTRIBUTING.md:103 require the full block on every source file.

Recommendation: Give all 35 the canonical header in the same pass that rewrites the 106 full headers, so the test tree does not stay as a third shape.

Review: confirmed. Reclassified all 85 .cpp: 50 Apache, 33 truncated box ('This file is part of' without copyright), 2 with nothing (test_engine_smoke.cpp, test_version.cpp). CONTRIBUTING.md:103-128 requires the full block.

#### L3. [medium] The whole `sa` CLI package has no licence header and no licence metadata

Evidence: 18 of 24 tracked .py files have no header: cli/sushiai/__init__.py, __main__.py, cli.py, config.py, console.py, env.py, services/{__init__,diag,discovery,project}.py and cli/tests/*.py (8 files); cli/sushiai/cli.py:1 opens directly with the module docstring. cli/pyproject.toml has no `license`, no `license-files`, no classifiers, and there is no LICENSE inside cli/, so a wheel or sdist built from cli/ ships with no licence at all.

Recommendation: Add the Python header to the 18 files, and add `license` (an SPDX expression or `LicenseRef-...` for a custom licence) plus `license-files` to cli/pyproject.toml, with the licence text reachable from cli/ at build time.

Review: confirmed. 18 of 24 .py carry no header (6 under tools/ do). cli/pyproject.toml read in full: no license, license-files or classifiers; git ls-files shows LICENSE only at the root, so setuptools finds no licence file in cli/.

#### L4. [medium] Three header shapes coexist for one licence

Evidence: Variant A: 22-line boxed Apache notice in 106 C++ files (include/SushiAI/SushiAI.h:1-22). Variant B: two '#' lines in 6 files under tools/ (tools/common/checker.py:1-2), which omit the file name, project name and URLs that variant A and the source-comments skill's block carry. Variant C: the truncated box in 33 tests (F2).

Recommendation: Decide one header per language for the new licence (owner decision: full notice or a short SPDX-style line) and apply it to all 147 C++ and Python source files in one commit.

Review: confirmed. Three shapes exist as described (106 / 6 / 33). The recommendation's '147 C++ and Python source files' is wrong: 147 leaves out the 18 headerless .py; the tree has 141 C++ + 24 Python = 165. One header per language applied in one pass is a uniform-sibling fix, so the SOLID shape is fine.

#### L5. [medium] 'Mustafa Garip & Sushi Systems' names two holders without saying who owns what

Evidence: The same string appears in LICENSE:178, 112 headers and docs/CONTRIBUTING.md:116. Commits come from personal addresses (mustafa.garip3434@hotmail.com, a GitHub noreply address); cli/pyproject.toml:8 uses mustafagarip@sushisystems.io. Nothing in the tree records an assignment between the person and the company.

Recommendation: Owner decision before the relicence: name the single licensor (the person or the legal entity) in the new licence text and headers. A non-commercial licence needs one identifiable party who can grant commercial exceptions.

Review: confirmed. Same holder string in LICENSE:178, headers and CONTRIBUTING.md:116; commit addresses are hotmail and GitHub noreply, pyproject.toml:9 (not :8) uses the sushisystems.io address. No assignment document is tracked.

#### L6. [medium] The contribution clause licenses inbound work under Apache-2.0 and grants the owner no right to relicense

Evidence: docs/CONTRIBUTING.md:12-13: 'By contributing you agree that your contributions are licensed under the project's **Apache License, Version 2.0**'. No CLA or DCO text exists in the tree.

Recommendation: Rewrite the clause together with the licence change. If outside contributions are to be accepted under a non-commercial licence, the owner needs an explicit assignment or a licence-back (CLA) so that commercial rights stay with Sushi Systems.

Review: confirmed. CONTRIBUTING.md:12-13 reads as quoted and no CLA/DCO exists. The audit omits CONTRIBUTING.md:6-7 ('external pull requests are not being accepted at this stage'), which makes the practical exposure nil today; low fits better than medium.

#### L7. [medium] Third-party components have no notice file, and the manifest does not record their licences or exact versions

Evidence: No NOTICE or third-party licence document is tracked. cli/sushistack.deps.toml:33-37 declares nlohmann-json with description, apt and vcpkg names but no licence and no version. cli/pyproject.toml:11-16 uses ranges (typer>=0.12, rich>=13.0, tomli>=2.0, sushicore>=0.7.0). The dependencies skill asks for the licence read from source and an exact version.

Recommendation: Add a third-party notice document listing nlohmann_json (MIT), GoogleTest (BSD-3-Clause, tests only), typer, rich, tomli (MIT) and the transitive hwloc and SYCL runtime. nlohmann_json is header-only and compiled into the sushiai library, so its MIT notice must accompany any binary distribution whatever the outbound licence is.

Review: confirmed. No NOTICE tracked; cli/sushistack.deps.toml:33-37 has no licence or version key; pyproject.toml:11-16 uses ranges. nlohmann/json is included only by src/config/value.cpp. Whether the deps fragment schema has a licence/version field at all is a sushicore question the audit did not raise.

#### L8. [medium] The dependency policy and the new outbound licence contradict each other across sibling repositories

Evidence: cli/sushistack.deps.toml:31 depends_on = ["sushiruntime", "sushiblas"]; cli/pyproject.toml:15 depends on sushicore>=0.7.0 from PyPI. The dependencies skill (C:/Users/sushi/.claude/skills/dependencies/SKILL.md) lists 'Non-commercial, source-available, field-of-use limits' as Rejected. Once the siblings move to the new licence, SushiAI depends on components the skill forbids.

Recommendation: Amend the skill's table to exempt first-party Sushi Systems modules, and relicense the siblings in a defined order. sushicore versions already on PyPI stay under the licence they were uploaded with.

Review: unclear. The evidence lines exist and the skill row reads as quoted, but the skill covers 'a third-party library, SDK or package'; first-party siblings are arguably outside its table already, so 'SushiAI depends on components the skill forbids' is an interpretation. The real conflict is narrower: anyone outside Sushi Systems applying the same table could not take Sushi modules. Editing a global skill is an owner decision, not a sushiai change.

#### L9. [medium] No checker verifies that a licence header exists or what it says, and the one that locates it is tied to the boxed shape

Evidence: D:/Projects/sushiai/tools/documentation/check_source_comments.py:29 (K_LICENSE_LINE = ^/\*.*\*/\s*$) and :54-67 only skip a leading run of boxed lines (C) or '#' lines (Python) to find where the body starts; no rule reports a missing block, a missing copyright line or the wrong licence name. That is why 33 truncated boxes and 20 headerless files pass today. The audit left this under not_checked. A short SPDX-style '//' header would also not match K_LICENSE_LINE, so the @file rule at :160 would misfire on every C++ file.

Recommendation: Add one licence-header rule to the checker (presence, holder line, licence identifier, one expected text per language held in a single constant) in the same task that picks the new header, and adjust the block locator if the header shape changes. Without it the relicence of 165 files cannot be verified mechanically.

Review: found by review.

#### L10. [medium] The audit does not say that the 2025 CC0 and MIT grants cover a different tree from the current one

Evidence: git show f06c42a:LICENSE is MIT (2025-07-02 to 2026-08-01); D:/Projects/sushiai/docs/design/ROADMAP.md:176-179 says the relicence to Apache-2.0 touched 'root LICENSE + README only, since its current tree is the retired prototype AI-0 deletes wholesale'. So the CC0 and MIT grants attach to the deleted prototype, and the present code was written after 7988fd6 under Apache-2.0. F1 lists the three licences without this split, which overstates what CC0 covers.

Recommendation: In the licence design document, state per period which tree each grant covers: prototype under CC0 then MIT, the AI-0 rewrite onward under Apache-2.0. Confirm with git log that no current file predates 7988fd6 before writing that sentence.

Review: found by review.

#### L11. [medium] Humanizer and scope check on the audit itself

Evidence: The prose is plain and carries no filler; findings state a fact and a fix. Two defects of substance: F11 tells the reader to 'remove or reword' the dated decision at docs/design/ROADMAP.md:180 while F13 says to supersede D7 with a new entry, and F7 rates the contribution clause medium without quoting docs/CONTRIBUTING.md:6-7, which says external pull requests are not accepted.

Recommendation: Keep F13's rule (supersede, do not edit D7) and drop the reword from F11. Lower F7 to low and quote lines 6-7 next to 12-13.

Review: found by review.

#### L12. [low] Copyright start year disagrees between LICENSE and the file headers

Evidence: LICENSE:178 'Copyright 2025-present Mustafa Garip & Sushi Systems'; all 112 headers and docs/CONTRIBUTING.md:116 say '2026-present'. First commit is dated 2025-06-08; docs/design/ROADMAP.md:179 says the 2025 prototype tree was deleted wholesale.

Recommendation: Pick one start year with the owner and use it in the new LICENSE and every header. 2026 fits the current code if none of the 2025 prototype survives; 2025 fits the repository.

Review: confirmed. LICENSE:178 says 2025-present; all 112 headers and CONTRIBUTING.md:116 say 2026-present; ROADMAP.md:179 says the prototype tree was deleted. The MIT text at f06c42a also used 2025-present.

#### L13. [low] Sole human author, but 101 commits carry AI co-author trailers

Evidence: git shortlog -sne HEAD lists only Mustafa Garip (117 + 9 commits under two addresses). Commit bodies carry 101 'Co-Authored-By: Claude ... <noreply@anthropic.com>' trailers.

Recommendation: No third-party human consent is needed to relicense. Record in the licence design document that the owner is the only contributor; the trailers do not name a rights holder but the owner may want that stated once.

Review: confirmed. shortlog recomputed: 117 + 9, one person; trailers recomputed: 54 + 39 + 6 + 1 + 1 = 101.

#### L14. [low] SushiBLAS carries portBLAS-derived code under Apache-2.0, and SushiAI links it

Evidence: docs/design/ROADMAP.md:169-172 (D5: 'portBLAS port goes ahead; attribution via NOTICE.md' in sushiblas), :196, :378-380, :672. No portBLAS code or attribution was found in this repository's tracked files.

Recommendation: No file change here. Apache-2.0 code can sit inside a non-commercial work as long as its notices are kept, so the sushiblas NOTICE.md must survive that repository's relicence. Remove or reword the ROADMAP sentence at :180 that justifies Apache-2.0 by 'same license family'.

Review: confirmed. ROADMAP.md:169-172, 196, 378-380, 672 read as cited; grep for portblas in tracked source returns nothing. The advice to 'remove or reword' ROADMAP.md:180 conflicts with F13, which rightly says to supersede D7 with a new decision; a dated decision record should not be edited.

#### L15. [low] No GPL, LGPL or proprietary SDK is declared, but the transitive runtime stack was not verified from here

Evidence: Direct dependencies are MIT or BSD-3-Clause (src/CMakeLists.txt:111, tests/CMakeLists.txt:28, cli/pyproject.toml:11-16). hwloc (BSD-3-Clause upstream) arrives through SushiRuntime (cmake/Vcpkg.cmake:3). The SYCL implementation is chosen by the workspace: cmake/SushiAISyclTarget.cmake:19 names AdaptiveCpp and oneAPI. Intel oneAPI DPC++ runtime libraries ship under Intel's own redistribution terms.

Recommendation: In the sushiruntime and sushistack audits, confirm which SYCL runtime DLLs the workspace redistributes and under what terms; that is where a vendor licence could impose conditions on a SushiAI binary.

Review: confirmed. find_package calls at src/CMakeLists.txt:111 and tests/CMakeLists.txt:28 exist; cmake/Vcpkg.cmake:1-5 names hwloc as transitive through SushiRuntime. Licence names come from upstream knowledge, as the audit says; nothing here contradicts them.

#### L16. [low] Agent and archive documents state Apache-2.0 as the rule and will go stale

Evidence: docs/agent/2026_09_16_CODE_STYLE_AND_COMMENTS/PLAN.md:16, REPORT.md:13, SPEC.md:22-33; docs/archive/superpowers/plans/2026-08-07-config-driven-training-plan.md:35; docs/design/ROADMAP.md:176-181 (D7).

Recommendation: Leave docs/archive/ and finished docs/agent/ reports untouched (frozen history). Add a new decision entry to docs/design/ROADMAP.md that supersedes D7, and update the live pages README.md:575-577 and docs/CONTRIBUTING.md:12-13, 103-128.

Review: confirmed. All cited lines exist (PLAN.md:16, REPORT.md:13, SPEC.md:22-33, archive plan :35, ROADMAP.md:176-181). The README statement is at :577 under the heading at :575.

#### L17. [low] Build and config files carry no licence line, and the installed package does not ship LICENSE

Evidence: 9 cmake/*.cmake, 6 CMakeLists.txt, cmake/SushiAIConfig.cmake.in, 4 .toml and .github/workflows/ci.yml have no header. CMakeLists.txt:103-105 and src/CMakeLists.txt:135-199 install targets, headers and package config but not LICENSE.

Recommendation: Owner decision whether build scripts get a one-line header. Add LICENSE (and the third-party notice) to the install rules so an installed SushiAI tree carries its terms; that is a build-configuration edit and needs its own task.

Review: confirmed. install() calls at CMakeLists.txt:103 and src/CMakeLists.txt:135, 143, 199 install no LICENSE; cmake, toml and yml files carry no header. The source-comments skill requires the block on source files only, so the header half is an owner choice, as the audit says.

### Not checked

- Whether the GitHub repository SushiSystems/SushiAI is public, and whether it has forks, GitHub releases or remote tags (no network query result was obtained; local tags are empty).
- Whether a package named sushiai-cli exists on PyPI; the tree gives no sign of an upload, but PyPI was not queried.
- The exact content of LICENSE at commit f06c42a (2025-07-02); only the initial CC0 text and the MIT-to-Apache commit subject were read.
- Third-party licence texts were not read from installed copies (nlohmann_json, GoogleTest, typer, rich, tomli, setuptools, hwloc, AdaptiveCpp/oneAPI); licences are stated from knowledge of the upstream projects.
- The licence, PyPI metadata and release history of sushicore, sushiruntime and sushiblas (other repositories, outside this scope).
- build/ and any other untracked or ignored directory, including whatever a build fetched or copied there.
- Whether any source file contains code copied from an outside project without attribution; only header blocks and licence keywords were searched, not code provenance.
- Full body text of the 34 Markdown files; they were keyword-searched (apache, licen[sc]e, copyright, SPDX, MIT, BSD, GPL), not read end to end.
- Whether tools/documentation/check_source_comments.py validates the licence text itself; the grep shows it only locates the block boundary (lines 29, 54-73), the rule bodies were not read in full.

## Command line

### Facts

- Entry points: console scripts `sa` and `sushiai`, both bound to `sushiai.cli:app` (D:/Projects/sushiai/cli/pyproject.toml:18-20); `python -m sushiai` through cli/sushiai/__main__.py. Distribution name `sushiai-cli`, version 0.1.0, requires-python >=3.10.
- Framework: Typer (`typer>=0.12`), one flat file cli/sushiai/cli.py (262 lines) plus services/{project,diag,discovery}.py; there is no commands/ package. Root app: no_args_is_help=True, add_completion=False, rich_markup_mode="rich", no @app.callback.
- sushicore requirement: `sushicore>=0.7.0` (cli/pyproject.toml:15); the sibling checkout D:/Projects/sushicore/pyproject.toml is at 0.7.0.
- Command tree, Project panel: `build` (--type/-t release|debug|relwithdebinfo, --clean, --define/-D VAR=VALUE repeatable); `test` (--suite/-s unit|regression|integration|functional|all|package, --filter/-f, --repeat/-r min 0); `run` ([TARGET], --sort, args after --); `clean` (no options); `doxygen` (no options).
- Command tree, Models panel: `demo mlp`, `train`, `inference`. All three set allow_extra_args, ignore_unknown_options and help_option_names=[] and forward every argument, --help included, to sushiai_demo_mlp / sushiai_train / sushiai_inference.
- Command tree, Diagnostics panel: `config` (no options), `env` (--all), and from sushicore `setup` (--dry-run, --yes, --toolchain repeatable, --no-gpu), `doctor` (--for GROUP), `link` (--workspace), `unlink` (--workspace).
- From sushicore: setup/doctor/link/unlink (provision.commands.register_provision_commands, cli.py:250-258, uses_base=True), help page (typer_help.help_group), console (cli_console.LazyConsole), config loading (ModuleConfig, StackConfig, ModuleProfile), build env (StackBuildEnv), config/env commands (diag.Diagnostics), executable search (discovery.ExecutableIndex), process spawning (proc.Runner), cmake/ctest/doxygen invocation (cmake_driver.CMakeDriver), toolchain_args, cmake_cache.
- Implemented locally: the build/test/run/clean/doxygen/demo/train/inference command declarations in cli.py, and the build policy in services/project.py (sibling check, configure args, the package-consumer suite, demo and app launchers). cli.py is line-for-line the same shape as D:/Projects/sushiblas/cli/sushiblas/cli.py for build/test/run/clean/doxygen/config/env.
- Help panels: three constants K_PROJECT="Project", K_MODELS="Models", K_DIAGNOSTICS="Diagnostics" (cli.py:21-23), pinned by cli/tests/test_help_screen.py K_GROUPS. Examples are carried in each command's `epilog` as `sa ...  # comment` lines.
- Output: sushicore Console through RichRenderer; `console.header/info/success/error` plus raw `console.console.print("[muted]...[/muted]")`. Child process output is inherited (Runner.run) or drained line by line with markup off (ctest). No --json, no --describe, no machine mode: sa never sets LazyConsole.machine.
- Version flag: none. `sa --version` is not declared anywhere in cli/sushiai; among the module CLIs only se declares one (D:/Projects/sushiengine/cli/sushiengine/cli.py:111-115).
- Exit codes: every command ends in `raise typer.Exit(<service return>)`. 0 success; 1 for missing build/, missing sibling, missing executable, missing Doxyfile or doxygen; 2 for a malformed -D entry, --filter/--repeat with --suite package, unknown demo, unknown doctor group, missing module in setup, no workspace in link/unlink; otherwise the child's own code (cmake, ctest, the launched binary).
- Errors: `console.error(...)` one line or a short bulleted block, then a non-zero return. Outside a checkout ModuleConfig.find_project_root raises SystemExit with the profile's "Not inside a SushiAI project..." message. Runner is built with defaults (services/project.py:90): missing_exit_code=1, catch_interrupt=False.
- Colour: `[cli] color = "auto"` in cli/config.toml:41-47, overridable by SUSHI_CLI_COLOR and by NO_COLOR (sushicore/config.py:108-113); auto means sys.stdout.isatty(). There is no --no-color flag. The logo is shown only on a UTF-8 terminal with 256 colours or more.
- Help wording register: imperative, sentence case, full stop ("Configure and build the project.", "Remove the build/ tree."). Same register as sr, sb, se; sd uses third person ("Configures (if needed) and builds every CMake target.").
- Config layering as coded: defaults -> cli/config.toml -> workspace-shared config (workspace.toml) -> cli/config.local.toml -> SA_* env, with SUSHIRUNTIME_DIR / SUSHIBLAS_DIR unprefixed (cli/sushiai/config.py:31-46, sushicore/module_config.py:127-146).
- Tests of the CLI surface: cli/tests/test_help_screen.py, test_provision_commands.py, test_console_lazy.py, test_config.py, test_demo.py, test_train.py, test_suites.py (not run; read-only task).

### Corrections from review

- F1 counts four stale docs/api-site passages; there are five. D:/Projects/sushiai/CLAUDE.md:10 also says 'Never edit generated Doxygen output under docs/api-site/', and that directory does not exist.
- F17's title says the default run target is stated three times; the evidence itself lists four places (cli/sushiai/config.py:37, :78, cli/config.toml:19, :30).
- F15 leaves open whether the intel/llvm step has its own prompt. It does not: sushicore/provision/steps.py:405-406 calls install_intel_llvm(cfg, dry_run, refresh=...) with no assume_yes, and intel_llvm.py has no confirm call. --yes covers only the AdaptiveCpp LLVM consent prompt (toolchains/adaptivecpp.py:39-77).
- F9 says a drained ctest child is 'left running' on Ctrl+C. On a console interrupt the child is in the same process group and receives the signal too; what the code skips is proc.terminate()/wait() and the 130 exit code (sushicore/proc.py:135-140).
- Recomputed and correct: cli.py is 262 lines; sushicore requirement is >=0.7.0 and D:/Projects/sushicore/pyproject.toml:7 is 0.7.0; the command tree and panel membership match cli.py:40-258 and test_help_screen.py:15-19; among sr/sb/sd/se/st only se declares --version.

### Findings

#### C1. [high] Documentation states the Doxygen output lands in docs/api-site/html; the Doxyfile writes it under build/docs/api-site

Evidence: D:/Projects/sushiai/Doxyfile:16 `OUTPUT_DIRECTORY = build/docs/api-site`, :48 `WARN_LOGFILE = build/docs/api-site/doxygen-warnings.log`; cli/sushiai/services/project.py:561 creates `build/docs`. Against that: docs/guides/CLI_GUIDE.md:55 and :425-427 ("Output lands in `docs/api-site/html/`, and warnings ... in `docs/api-site/doxygen-warnings.log`"), README.md:226, docs/getting_started/TUTORIAL.md:487. A consequence the guide never states: `sa clean` removes the generated reference with the build tree.

Recommendation: Correct the four passages to build/docs/api-site, and say in the `sa clean` section that it also removes the API reference. sd's help string says docs/api-site/html (sushidsp/cli/sushidsp/cli.py:132), so pick one output location for every module and make each Doxyfile and guide agree.

Review: confirmed. Doxyfile:16 and :48 write under build/docs/api-site; CLI_GUIDE.md:55 and :425-426, README.md:226 and TUTORIAL.md:487 say docs/api-site; no docs/api-site exists. The audit missed a fifth passage, CLAUDE.md:10.

#### C2. [high] A malformed cli/config.toml or config.local.toml reaches the user as a traceback

Evidence: D:/Projects/sushicore/sushicore/config.py:55-59 `_read_toml` calls `tomllib.load` with no handler, and it runs on the first console use (`console.header(...)` at cli/sushiai/services/project.py:192, 248, 459 and so on). sushicore/workspace.py:42 `read_toml` is equally unguarded and feeds `sa config`. Nothing in cli/sushiai/cli.py or the services catches TOMLDecodeError; the only handler is for link editing (workspace.py:238). The help page alone survives, because typer_help._write_guarded catches Exception.

Recommendation: Catch TOMLDecodeError once in sushicore's two TOML readers and re-raise a config error that names the file and line; have each CLI entry point turn that error into one console.error line and exit 2. Fix it in sushicore so all seven CLIs get it.

Review: confirmed. sushicore/config.py:55-59 and workspace.py:42-47 call tomllib.load unguarded; config_base.py:21 loads through the same read_toml; cli.py and the services have no handler. Fix is brick-shaped (own error at the sushicore boundary, one line at the entry point), matching python-code-style Errors.

#### C3. [medium] `sa train --help`, `sa inference --help` and `sa demo mlp --help` print no help without a checkout and a finished build

Evidence: cli/sushiai/cli.py:159-163, 185-189, 206-210 set `help_option_names: []`, so --help is forwarded. services/project.py:529-539 and 496-506 then need find_project_root, build/ and the built binary; otherwise the user gets "build/ not found. Run `sa build` first." and exit 1. These commands also carry no epilog examples, unlike every other sa command (test_help_screen.py:22 excludes them on purpose). README.md:230-231 says "each command's own help ends with examples".

Recommendation: Keep forwarding, but when the binary is absent print the Typer docstring page (usage, positional arguments, one example) before the build hint. Add epilog examples for train and inference so the root page and the README sentence hold.

Review: confirmed. cli.py:159-163, 185-189, 206-210 set help_option_names to []; project.py:496-506 and 529-539 need a checkout, build/ and the binary first. test_help_screen.py:22 excludes train and inference; README.md:230-232 claims every command's help ends with examples.

#### C4. [medium] `train` and `inference` help text names the program `sushiai` while every other command and example says `sa`

Evidence: cli/sushiai/cli.py:194 "Usage: sushiai train <train.json>", :197 "Example: sushiai train examples/heavy_mlp/train.json", :215 "Usage: sushiai inference <model.json> <model.ckpt> <eval.json>". The app is `name="sa"` (cli.py:28) and all epilogs use `sa`. docs/guides/CLI_GUIDE.md:57-58 and :317-328 use `sushiai train` in a table whose other rows are `sa ...`; README.md:231 writes `sa train`.

Recommendation: Use `sa train` / `sa inference` in the help strings, the guide's overview table and its section heading, and keep one sentence saying `sushiai` is the long alias.

Review: confirmed. cli.py:194, 197, 215 say `sushiai`, the app is name="sa" (cli.py:28), README.md:231 says `sa train`. CLI_GUIDE.md:326-328 states the `sushiai` spelling is deliberate, so the guide half is an owner choice, not a slip; the audit did not say so.

#### C5. [medium] `sa config` cannot report config.local.toml or the workspace file as a source, though the guide says it does

Evidence: docs/guides/CLI_GUIDE.md:448-450: "which source (default / `config.toml` / `config.local.toml` / an `SA_*` environment variable) supplied each field". D:/Projects/sushicore/sushicore/diag.py:85-90 returns only `env:<VAR>`, `config.toml` or `default`; lines 75-83 merge both local files into one key set, and the workspace-shared file from module_config.py:127-135 is not read at all, so a value `hub install` wrote shows as `default`.

Recommendation: Make Diagnostics._source_of report the actual layer (config.toml, workspace.toml, config.local.toml, env) in sushicore, then keep the guide sentence. Until then, shorten the guide to the three values the command prints.

Review: confirmed. sushicore/diag.py:85-90 returns only env:<VAR>, config.toml or default; :75-83 merges both repo files into one key set and never reads the workspace file that module_config.py:127-135 layers. CLI_GUIDE.md:448-450 promises config.local.toml as a source.

#### C6. [medium] The same action has different names and shapes across the sibling CLIs

Evidence: Picker: `sa run --sort` (cli/sushiai/cli.py:113-114) against `se run --pick`, where --sort is a hidden deprecated alias (sushiengine/cli/sushiengine/cli.py:251-255). API reference: `sa doxygen` against `se docs` (sushiengine cli.py:293-302). Skipping tests: sr and se have `build --no-test` (sushiruntime/cli/sushiruntime/cli.py:103-105, sushiengine cli.py:165-168); sa has only `-D SA_BUILD_TESTS=OFF`. `test`: se and sd build first (se `--no-build`, sushiengine cli.py:214-217); sa fails with "build/ not found" (services/project.py:252-254). `env --all` has `-a` in sd (sushidsp/cli/sushidsp/cli.py:120) and not in sa (cli.py:243-244). `run`: se adds --list and --no-build; sa has neither.

Recommendation: Fix one spelling per action in sushicore and register it from there, as setup/doctor/link/unlink already are: --pick (with --sort as a hidden alias), one name for the API-reference command, --no-test on build, one rule for whether test builds first, -a on env --all. This is a public interface decision for the owner.

Review: confirmed. Checked sa cli.py:113-114 (--sort), se cli.py:252-254 (--pick, hidden --sort), se :294 (docs), sr cli.py:104 and se :166 (--no-test), se :215 (--no-build), se :263 (--list), sd cli.py:99 (test builds first) and :120 (-a). All as stated.

#### C7. [medium] Help panel names differ per CLI, and the same command sits under a different heading in each

Evidence: sa: Project / Models / Diagnostics (cli/sushiai/cli.py:21-23). sb: Project / Documentation / Diagnostics, with doxygen under Documentation (sushiblas/cli/sushiblas/cli.py:21-23, 134). sd: Build / Application / Environment, with config, env and the provision commands under Environment (sushidsp/cli/sushidsp/cli.py:23-25, 175). se uses K_BUILD, K_ENVIRONMENT and others. `doxygen` is in Project for sa and sr, Documentation for sb, Build for sd.

Recommendation: Define the shared panel names once in sushicore (for example Build, Diagnostics, Environment) and have each CLI import them; a module adds only its own panel (Models here).

Review: confirmed. sa cli.py:21-23, sb cli.py:21-23 with doxygen at :134 under Documentation, sd cli.py:23-25 with doxygen at :128 under Build, sr cli.py:26-29 with doxygen at :207 under Project, se cli.py:41-45.

#### C8. [medium] No `--version`, and none of the machine-output flags hub offers

Evidence: cli/sushiai/cli.py declares no @app.callback and no --version; a search of cli/sushiai for "--version", "--json", "--describe" returns nothing. se has --version (sushiengine/cli/sushiengine/cli.py:111-115); hub has --json and --describe (sushistack/cli/sushihub/cli.py:54-57). sushicore's LazyConsole.machine switch (cli_console.py:39-49) is never set by sa.

Recommendation: Add one root callback in sushicore (version, --json, --describe) that every module CLI registers, so `sa --version` prints the CLI version, the sushicore version and the project root in the same shape as `se --version`.

Review: confirmed. No callback, --version, --json or --describe in cli/sushiai; se cli.py:114 and sushihub/cli.py:54-57 have them; LazyConsole.machine (cli_console.py:39-49) is never set. SOLID note: the recommended single callback carries three duties (version, JSON mode, describe); three registrable options would be the brick shape.

#### C9. [medium] Ctrl+C during a build or test is not handled, and a drained ctest child is left running

Evidence: cli/sushiai/services/project.py:90 `Runner(console, PROFILE.program)` keeps `catch_interrupt=False`. D:/Projects/sushicore/sushicore/proc.py:100-103 and :135-140 then re-raise KeyboardInterrupt; in run_drained the `proc.terminate()` at line 138 is skipped. Click turns the exception into "Aborted!" with exit 1, where sushitrack passes catch_interrupt=True and returns 130 (sushitrack/cli/sushitrack_cli/process.py:25-26).

Recommendation: Pass catch_interrupt=True in sa (and sr, sb, se, sd), or make it the default in sushicore.proc.Runner, so an interrupt prints one warning, stops the child and exits 130 everywhere.

Review: confirmed. project.py:90 builds Runner with defaults; proc.py:100-103 and :135-140 re-raise; sushitrack process.py:26 passes catch_interrupt=True. 'Left running' is overstated: a console Ctrl+C also reaches the child, so the real defect is exit 1 with 'Aborted!' instead of 130, and no terminate on non-console interrupts.

#### C10. [medium] Filesystem failures in the package suite and in clean are either a traceback or silent

Evidence: cli/sushiai/services/project.py:309-310 `shutil.copy2(dll, dest / dll.name)` has no handler (a locked DLL on Windows raises PermissionError to the user). Lines 215 and 388-390 use `shutil.rmtree(..., ignore_errors=True)`, and sushicore/cmake_driver.py:138-139 does the same and then prints "<build> removed." whether or not the tree is gone; the suite's own comment (project.py:382-387) says a stale directory makes the run prove nothing.

Recommendation: Report an rmtree that leaves the directory behind as an error with the path, and catch OSError around the DLL staging with a one-line message. The clean_tree half belongs in sushicore.

Review: confirmed. project.py:309-310 copy2 without a handler; :215 and :388-390 rmtree with ignore_errors=True; cmake_driver.py:138-139 prints 'removed' unconditionally. Line numbers match.

#### C11. [medium] `sa build --clean -D BAD` deletes the build tree and then rejects the -D entry

Evidence: D:/Projects/sushiai/cli/sushiai/services/project.py:201-202 runs _DRIVER.clean_tree and :214-215 runs shutil.rmtree on a fresh configure, both before _configure_args validates the entries at :218-222 (the check itself is at :180-184). A typo such as `-D SA_BUILD_TESTS` exits 2 after the tree is gone. docs/guides/CLI_GUIDE.md:93-94 says only that the entry 'is rejected before configure runs', which hides the loss.

Recommendation: Validate the defines in their own function before any filesystem change (parse first, then clean, then configure), and return 2 with the tree untouched. sb's build has the same shape (sushiblas/cli/sushiblas/cli.py:55), so the parser belongs in sushicore next to toolchain_args.

Review: found by review.

#### C12. [medium] The CLI package does not follow the python-code-style layout the brief names as the yardstick, and the audit moved this to another dimension

Evidence: python-code-style: 'commands/ parses arguments, services/ does the work. A service never imports the command framework or writes to the terminal', dependencies arrive through __init__, enum members are UPPER_SNAKE. D:/Projects/sushiai/cli/sushiai has no commands/ package (all commands in cli.py:40-258); services/project.py prints through console on nearly every path (:102, :192, :204, :253, :474) and holds module-level singletons _RUNNER and _DRIVER (:90-91); BuildType and Suite members are lower case (:38-56). The audit lists this only under not_checked.

Recommendation: Treat it as a CLI finding, since it decides how a shared command layer in sushicore can be registered: services return a result (code plus messages) and a commands/ layer renders it. Module boundary and public shape are owner decisions; ask before moving anything.

Review: found by review.

#### C13. [medium] The guide's flag list for `sa demo mlp` was never compared with the binary, though the guide and the Typer docstring both send the reader there

Evidence: The audit lists this under not_checked. A spot check: D:/Projects/sushiai/examples/demo_mlp.cpp:357-406 accepts --epochs, --batches, --batch-size, --hidden, --features, --classes, --lr, --seed, --mnist-dir, --no-fuse, --fusion, --loss-scale, --precision, --optimizer, --activation; every flag used in docs/guides/CLI_GUIDE.md:225-232 is in that set, so no mismatch was found for demo mlp. apps/train.cpp and apps/inference.cpp were not compared against CLI_GUIDE.md:317-411.

Recommendation: Close the gap for train and inference (argument order, exit codes, the `compiles: 1` line) before the guide is called accurate; the demo half checks out.

Review: found by review.

#### C14. [low] `demo()` and `run_app()` are two copies of one launcher

Evidence: cli/sushiai/services/project.py:496-511 and :529-544 are the same sixteen lines (root, config, build-dir check, match_by_name, env, print, run). The run_app docstring at :517-520 says the shape was "factored out rather than repeated", and `demo` was not moved onto it.

Recommendation: Have demo() resolve the target name and call run_app(target, title, args). The launcher (find a built binary, run it from the repo root with the build env) is the same in sb, sr and se and is a candidate for sushicore.

Review: confirmed. project.py:496-511 and :529-544 are the same sequence; the run_app docstring at :517-520 claims it was factored out. Recommendation (demo resolves the name, then calls run_app) is brick-shaped.

#### C15. [low] Dead code in the CLI package

Evidence: cli/sushiai/services/discovery.py:21 `find_executables` and :26-28 `select_interactive` have no caller in cli/ (search of D:/Projects/sushiai/cli returns only their definitions). services/project.py:490-494 the unknown-demo branch cannot be reached: Typer only registers `mlp` (cli.py:153-171). project.py:559-560 falls back to `docs/Doxyfile`, which does not exist; the Doxyfile is at the repo root.

Recommendation: Delete the two unused discovery names, the docs/Doxyfile fallback, and either register demos from _DEMO_TARGETS or drop the unknown-name branch.

Review: confirmed. find_executables and select_interactive have no caller anywhere in the sushiai tree; no docs/Doxyfile exists. The unknown-demo branch (project.py:490-494) is unreachable from Typer but is a guard on a service function taking a free name, so 'dead' is too strong for that one item.

#### C16. [low] Three places still describe sb as modelling an older oneAPI build

Evidence: docs/guides/CLI_GUIDE.md:8-10 ("not SushiBLAS's `sb` (which still models an older, pre-toolchain-refresh build)"), cli/config.toml:12-15, cli/sushiai/services/project.py:8-10. sb now uses the same StackConfig and StackBuildEnv (sushiblas/cli/sushiblas/config.py:24, 55; env.py:18-23) and its cli.py matches sa's command for command.

Recommendation: Remove the comparison from all three; the guide can say the build commands are the ones sushicore's module CLIs share.

Review: confirmed. CLI_GUIDE.md:8-10, cli/config.toml:12-15 and project.py:8-10 carry the claim; sushiblas config.py:24, :55 and env.py:18-23 use StackConfig and StackBuildEnv.

#### C17. [low] Documented config precedence does not match the loader

Evidence: cli/config.toml:4 lists "... -> SA_* env vars -> CLI flags" although no sa flag overrides a config field, and it omits the workspace-shared layer; cli/sushiai/config.py:3-5 omits that layer too. sushicore/module_config.py:137-146 loads config.toml -> workspace config -> config.local.toml -> environment. docs/guides/CLI_GUIDE.md:40-48 names only the two repo files.

Recommendation: State the four real layers in config.toml's header, config.py's docstring and the guide's "How the CLI finds your compiler" section, and drop "CLI flags".

Review: confirmed. cli/config.toml:4 lists 'CLI flags' and omits the workspace layer; cli/sushiai/config.py:5 omits it too; module_config.py:127-146 loads four layers.

#### C18. [low] The guide's description of `sa setup --yes` names a different download than the code

Evidence: docs/guides/CLI_GUIDE.md:466 "`--yes` skips the confirmation prompt before the intel/llvm download". The flag becomes `assume_acpp_llvm` (sushicore/provision/commands.py:216) and is read only when installing AdaptiveCpp (sushicore/provision/steps.py:414-416); the help string says "the LLVM-download prompt" (commands.py:265-266). I did not trace whether the intel/llvm step has a prompt of its own.

Recommendation: Confirm which prompt --yes covers and write that in both the sushicore help string and the guide.

Review: confirmed. commands.py:216 maps --yes to assume_acpp_llvm, read only at steps.py:414-416 for AdaptiveCpp. The audit's open question resolves: install_intel_llvm (intel_llvm.py:24, called at steps.py:405-406) takes no assume_yes and has no confirm, so CLI_GUIDE.md:466 is wrong, not merely unverified.

#### C19. [low] Internal roadmap codes appear in user-facing help

Evidence: cli/sushiai/cli.py:166 demo mlp help: "Train an MLP end to end and report the compile count (AI-7 showcase)."; docs/guides/CLI_GUIDE.md:222 heading "`sa demo` — the AI-7 showcase". Source comments at cli.py:141 ("per ST-2") and services/project.py:9 ("pre-SB-0/SB-2") carry the same codes.

Recommendation: Drop the codes from help strings and headings; a reader of `sa demo --help` has no key for them.

Review: confirmed. cli.py:166 and :141, CLI_GUIDE.md:222, project.py:9 carry AI-7, ST-2, SB-0/SB-2.

#### C20. [low] The default run target is stated three times, one of them with a platform suffix

Evidence: cli/sushiai/config.py:37 `default_target="sushiai_example"`, :78 `target_bin: str = "sushiai_example"`, cli/config.toml:19 and :30 (`sushiai_example.exe` for Windows). ExecutableIndex.match already matches by stem (sushicore/discovery.py:107), so the .exe override adds nothing. services/project.py:561 also spells `root / "build"` directly beside `_build_dir` (:86-87).

Recommendation: Keep the target in PROFILE only, remove the Windows override, and use _build_dir in doxygen().

Review: confirmed. config.py:37 and :78, config.toml:19 and :30; sushicore/discovery.py:107 matches by stem; project.py:561 spells root / "build" beside _build_dir. The title says three places, the evidence lists four.

#### C21. [low] Stale pointers to env.py for the PATH mechanism

Evidence: docs/guides/CLI_GUIDE.md:186 "`sa` puts the vcpkg installed `bin` there, see `cli/sushiai/env.py`" and services/project.py:318 "(see env.py)". cli/sushiai/env.py is a 28-line wrapper; the PATH logic is in D:/Projects/sushicore/sushicore/build_env.py.

Recommendation: Point both at sushicore.build_env.StackBuildEnv.

Review: confirmed. CLI_GUIDE.md:186 and project.py:318 point at env.py, which is a 28-line wrapper over sushicore.build_env.StackBuildEnv.

#### C22. [low] Package metadata lags the command set and the Python floor

Evidence: cli/pyproject.toml:8 description "SushiAI developer CLI (build / test / run)" (the CLI also trains, runs inference, provisions and diagnoses); :10 `requires-python = ">=3.10"` with a tomli fallback at :14, while the python-code-style skill sets 3.11 or later.

Recommendation: Update the description; decide the Python floor once for all Sushi CLIs and sushicore and drop tomli if it is 3.11.

Review: confirmed. cli/pyproject.toml:8, :10, :14 as quoted; python-code-style says 3.11 or later. All six sibling CLI pyprojects also say >=3.10, so the floor is a cross-repository decision, as the recommendation says.

### Not checked

- Nothing was executed: no `sa` invocation, no pytest, no help page rendered. Every statement about output, exit codes and help layout comes from reading source.
- sushicore internals beyond what sa calls directly: provision/pipeline.py, steps.py, checks.py, doctor.py rendering, selection.py, closure.py, sinks.py, build_env.py body, stack_config.py, config_base.py body, console.py, renderer.py, theme.py, ui/ and help/ packages. Exceptions that could escape from those into `sa setup` / `sa doctor` were not traced.
- Whether `sa doctor --for GROUP` behaves as CLI_GUIDE.md:478 describes beyond doctor.py:69-71, and the list of valid group names.
- The C++ programs behind the forwarded commands (examples/demo_mlp.cpp, apps/train.cpp, apps/inference.cpp): their flag lists, their own --help text and exit codes, and whether the flags documented in CLI_GUIDE.md:224-298 and README match them.
- hub (sushistack/cli/sushihub), st (sushitrack) and sr/se/sd beyond their cli.py command declarations; service layers of the sibling CLIs were not read. `hub install-cli` was confirmed to exist by name only.
- cli/tests/*.py other than test_help_screen.py were not read; whether the tests pass is unknown.
- docs/reference/API_REFERENCE.md, docs/architecture/ARCHITECTURE.md, docs/guides/INTEGRATION.md, docs/CONTRIBUTING.md body, docs/design/*, docs/agent/* and docs/archive/* were not read for CLI statements; only grep hits for `sa <command>` in README, TUTORIAL, CONTRIBUTING, AGENTS.md, CLAUDE.md and KNOWN_ISSUES were checked.
- The README anchors and relative links cited from CLI_GUIDE.md were not resolved.
- Source-comment and licence-header shape of cli/sushiai/*.py (no licence block, no @file header, separator comment lines in cli.py:37-39) and the python-code-style layout rules (no commands/ package, services printing through the console) belong to the code and licence dimensions and were noted only in passing.
- Linux and macOS behaviour of the Windows-specific branches (DLL staging, vcvars) was not examined; cli/config.toml has no [tool.darwin] table and that was not followed further.
- sushi-module.toml and its relation to the CLI were not read.

## Documentation

### Facts

- Present: README.md (578 lines), docs/README.md, docs/CONTRIBUTING.md, docs/DOCUMENTATION_STYLE_GUIDE.md, docs/getting_started/ (TUTORIAL.md), docs/architecture/ (ARCHITECTURE.md), docs/guides/ (CLI_GUIDE.md, INTEGRATION.md), docs/reference/ with CHANGELOG.md, GLOSSARY.md, KNOWN_ISSUES.md, API_REFERENCE.md.
- Present: docs/design/ with README.md, REMAINING_WORK.md and one design document, ROADMAP.md (770 lines, '**Status:** Open — phase 2 of 4' on line 3, mapped in docs/design/README.md).
- Present: docs/agent/ with two conforming work folders (2026_09_16_CODE_STYLE_AND_COMMENTS, 2026_09_16_DOCS_AND_ROOT_HYGIENE), each holding SPEC.md, PLAN.md, REPORT.md. docs/archive/ exists.
- Missing: docs/modules/ (named in the global CLAUDE.md tree; the documentation skill's tree does not list it). docs/agent/specs and docs/agent/plans do not exist; the repository uses the skill's per-work-folder shape instead of the specs|plans|reports shape in the global CLAUDE.md.
- Missing module READMEs: no cli/README.md, apps/README.md, tests/README.md, examples/README.md, and none under include/SushiAI/<module>/ or src/<module>/ (core, graph, autograd, nn, optim, data, io, config, train). Only tools/README.md and examples/heavy_mlp/README.md exist.
- No place in the architecture: docs/agent/reports/ (2 loose dated reports); docs/archive/superpowers/plans and specs (5 files, lower-kebab names, 9 258 lines). By the placement questions the reports belong in a work folder as REPORT.md; the superpowers files are agent output from 2026-08-07 and belong in dated work folders under docs/archive/agent/.
- docs/reference/ also holds CODE_OF_CONDUCT.md and SECURITY.md, which the architecture does not name; docs/DOCUMENTATION_STYLE_GUIDE.md line 16 lists them as reference content, so the repository has decided this locally.
- Root carries AGENTS.md and CLAUDE.md, one LICENSE (Apache-2.0), one CHANGELOG (docs/reference/CHANGELOG.md), root Doxyfile. No duplicated LICENSE or CHANGELOG. No docs/slop or docs/superpowers at top level.
- Untracked, gitignored .superpowers/sdd/ exists at the repository root (one entry, 2026-08-07-cv-ir-autograd-plan).
- CHANGELOG.md: 64 entries under a single '## Unreleased' heading, no release section, no git tags; no line exceeds 240 characters; no nested bullets.
- Link check over 34 tracked Markdown files: one broken relative link (examples/heavy_mlp/README.md:101). All links in docs/README.md resolve.
- Versions agree at 0.1.0 in CMakeLists.txt:16 and cli/pyproject.toml:7; cli/pyproject.toml:15 requires sushicore>=0.7.0.

### Corrections from review

- CHANGELOG entry count: the audit says 64 entries under '## Unreleased'. `grep -c '^- 20' docs/reference/CHANGELOG.md` gives 62 (69 lines, 65 non-blank). The conclusions about length and nesting still hold: no line over 240 characters, no nested bullet.
- CHANGELOG line numbers in F15 and F4 are off by two: the reason clause is at line 10 (audit: 8), the PyPI entry at line 11 (audit: 9), the blank line inside the list at line 14 (audit: 13).
- F3 says 'four pages' carry docs/api-site; five do: README.md:226, docs/guides/CLI_GUIDE.md:55 and :425-426, docs/getting_started/TUTORIAL.md:487, CLAUDE.md:10, plus the .gitignore comment at :22.
- API_REFERENCE.md stale citations start at line 11 (docs/README.md), with docs/ARCHITECTURE.md at :12 and docs/INTEGRATION.md at :13; the audit gives :12-13 for all three.
- Recomputed and correct: 34 tracked Markdown files; README.md 578 lines; ROADMAP.md 770 lines; docs/archive/superpowers 5 files totalling 9 258 lines; version 0.1.0 in CMakeLists.txt:16 and cli/pyproject.toml:7; sushicore>=0.7.0 at cli/pyproject.toml:15; no git tags; 13 documents under docs/agent and docs/archive.

### Findings

#### D1. [high] GLOSSARY.md describes code that does not exist and misdefines SACP

Evidence: D:/Projects/sushiai/docs/reference/GLOSSARY.md: 'SushiAI::ir namespace (ir::Graph, ir::Node, ir::Value)', 'MatMulNode, ReluNode, CrossEntropyLossNode', 'Compiler::compile', and 'SACP (Sushi Autograd & Compilation Pipeline) ... constant folding, dead-code elimination'. grep over include/ and src/ for 'namespace ir|MatMulNode|ReluNode|Compiler::compile|constant fold|dead.code' returns 0 hits. include/SushiAI/io/checkpoint_format.hpp:44 says SACP is 'The first four bytes of every checkpoint: the ASCII "SACP"'; README.md calls it the checkpoint format. The real namespace is SushiAI::Graph (API_REFERENCE.md:22).

Recommendation: Rewrite the glossary from the headers: SushiAI::Graph names, the real op identities in core/ops.hpp, SACP as the checkpoint format, and drop passes the library does not have.

Review: confirmed. Read docs/reference/GLOSSARY.md: it names SushiAI::ir, MatMulNode, ReluNode, Compiler::compile and defines SACP as a compiler pipeline. grep over include/ and src/ returns no hit; include/SushiAI/io/checkpoint_format.hpp:44 gives SACP as the checkpoint magic; API_REFERENCE.md:22 uses SushiAI::Graph::Graph.

#### D2. [high] KNOWN_ISSUES.md contradicts the code and links to anchors that do not exist

Evidence: D:/Projects/sushiai/docs/reference/KNOWN_ISSUES.md issue 2 says the allocator 'assigns distinct device memory slots to intermediate activations rather than calculating lifetime overlaps'; include/SushiAI/graph/memory_plan.hpp and arena.hpp exist and README.md:47-49 plus ARCHITECTURE.md §7 say IR values are packed by lifetime and only lowering scratch is unpacked. Issue 5 describes a host-to-device staging copy with 'pinned memory or asynchronous DMA'; ARCHITECTURE.md §7 describes two device-side blas().copy() calls. Five Tracking links use anchors (#1-whole-step-loss-scaling-skip, #2-scratch-memory-arena-packing, #3-row-wise-log-softmax-fusion, #4-standalone-log-softmax-backward-ir-node, #6-batch-ingestion-host-copy-overhead) but REMAINING_WORK.md has no numbered headings, only five '## ' group headings.

Recommendation: Rewrite issues 2, 4 and 5 from ARCHITECTURE.md §7 and point each Tracking link at a heading that exists in REMAINING_WORK.md.

Review: confirmed. Read KNOWN_ISSUES.md and REMAINING_WORK.md. Five Tracking links carry numbered anchors (#1, #2, #3, #4, #6); REMAINING_WORK.md has five unnumbered '## ' group headings and bullets, so none resolves. Issue 2 and 5 text contradicts REMAINING_WORK.md:7-8 (planner packs IR values; batch feed is two blas().copy() calls). Issue 4 also contradicts REMAINING_WORK.md:14 (standalone LOG_SOFTMAX autodiff does not work at all, it is not 'decomposed').

#### D3. [high] Doxygen output path is documented as docs/api-site but the Doxyfile writes to build/docs/api-site

Evidence: D:/Projects/sushiai/Doxyfile:16 'OUTPUT_DIRECTORY = build/docs/api-site', :48 WARN_LOGFILE under build/. Contradicted by README.md ('sa doxygen # build the Doxygen API reference into docs/api-site/html/'), docs/guides/CLI_GUIDE.md:55 and :422-426 ('Output lands in docs/api-site/html/ ... docs/api-site/doxygen-warnings.log'), docs/getting_started/TUTORIAL.md:487, CLAUDE.md ('generated Doxygen output under docs/api-site/'). .gitignore:22 also says 'Doxyfile is committed at docs/Doxyfile'; it is at the root.

Recommendation: Correct the four pages and the .gitignore comment to build/docs/api-site, in the commit that owns the Doxyfile.

Review: confirmed. Doxyfile:16 and :48 write under build/docs/api-site. docs/api-site appears in README.md:226, CLI_GUIDE.md:55 and :425-426, TUTORIAL.md:487, CLAUDE.md:10. .gitignore:22 says 'Doxyfile is committed at docs/Doxyfile'; the file is at the root. That is five pages, not four, counting CLAUDE.md. The audit did not note that cli/sushiai/services/project.py:558-560 still falls back to docs/Doxyfile, and .gitignore:23-25 still ignores a docs/api-site that nothing writes.

#### D4. [high] CLI_GUIDE.md install section says sushicore is on no index; it is a PyPI dependency

Evidence: D:/Projects/sushiai/docs/guides/CLI_GUIDE.md:26-30: 'injects the shared sushicore package — which is published to no index, so pipx cannot resolve it as an ordinary dependency'. cli/pyproject.toml:15 declares 'sushicore>=0.7.0'; docs/reference/CHANGELOG.md:9 'Took sushicore from PyPI instead of a sibling checkout'. The guide's opening paragraph (lines 7-10) also describes the CLI as modelled on se and 'not SushiBLAS's sb (which still models an older, pre-toolchain-refresh build)', a claim about another repository's state that this page cannot keep true.

Recommendation: Rewrite 'Installing the CLI' against the current pyproject and the hub install path; remove the cross-repository comparison.

Review: confirmed. CLI_GUIDE.md:26-28 says sushicore 'is published to no index'; cli/pyproject.toml:15 requires sushicore>=0.7.0 and CHANGELOG.md:11 records the move to PyPI. Checked what the audit left open: D:/Projects/sushistack/cli/sushihub/services/cli_install.py:12 says the install 'stops there: sushicore is an ordinary PyPI' dependency, so the inject step the guide describes no longer exists. Cross-repository comparison is at CLI_GUIDE.md:7-10. The changelog line cited is :11, not :9.

#### D5. [medium] REMAINING_WORK.md cites three paths that do not exist

Evidence: D:/Projects/sushiai/docs/design/REMAINING_WORK.md:7 `src/graph/memory_planner.cpp` (actual: src/graph/memory_plan.cpp); :18 `include/SushiAI/train/loss_scaler.hpp` (actual: include/SushiAI/optim/loss_scaler.hpp); :28 and :29 `cli/sushiai/commands/demo.py` (no cli/sushiai/commands/ folder; demo lives in cli/sushiai/cli.py and cli/sushiai/services/project.py).

Recommendation: Correct the four citations.

Review: confirmed. src/graph/memory_plan.cpp exists, memory_planner.cpp does not; loss_scaler.hpp is under include/SushiAI/optim/, include/SushiAI/train/ holds only objective.hpp and trainer.hpp; cli/sushiai/commands/ does not exist. The title says three paths and the recommendation says four citations; both are right (three distinct paths, four occurrences).

#### D6. [medium] More than one backlog: ARCHITECTURE.md §7 and KNOWN_ISSUES.md restate REMAINING_WORK.md

Evidence: D:/Projects/sushiai/docs/architecture/ARCHITECTURE.md:867-939 '## 7. What's not here yet' lists nine open items (scratch packing, LOG_SOFTMAX fusion, batch feed copies, standalone LOG_SOFTMAX rule, per-parameter skip, bfloat16, Dataset seam, Trainer leaves, AI-7 gates), the same items as docs/design/REMAINING_WORK.md. docs/reference/KNOWN_ISSUES.md items 1-5 repeat five of them. docs/design/ROADMAP.md:597 'Not landed:' and §24 'Open questions' carry a third copy. README.md:17 sends readers to the §7 list, not to the backlog. The copies already disagree (F2).

Recommendation: Keep open work in REMAINING_WORK.md only; reduce §7 to a link, keep KNOWN_ISSUES.md to user-visible behaviour with workarounds.

Review: confirmed. ARCHITECTURE.md:867 '## 7. What's not here yet' opens a bullet list of open items; KNOWN_ISSUES.md items 1-5 restate backlog items; ROADMAP.md:597 'Not landed:' and :668 '## 24. Open questions' hold more; README.md:18 links to the §7 list. The skill allows one backlog. The recommendation is brick-shaped: one owner (REMAINING_WORK.md), the others link.

#### D7. [medium] docs/agent/reports/ holds loose reports outside any work folder

Evidence: D:/Projects/sushiai/docs/agent/reports/2026-09-22-help-screen-task-27.md and 2026-09-22-wave-1-task-7c.md. The documentation skill and docs/DOCUMENTATION_STYLE_GUIDE.md:28-29 allow only YYYY_MM_DD_WORK_NAME folders holding SPEC.md, PLAN.md, REPORT.md. Names are lower-kebab, not UPPER_SNAKE_CASE.

Recommendation: Fold each into a dated work folder as REPORT.md (owner approval needed for the move), or decide stack-wide that docs/agent/reports is the shape and change the skill.

Review: confirmed. docs/agent/reports/ holds 2026-09-22-help-screen-task-27.md and 2026-09-22-wave-1-task-7c.md. DOCUMENTATION_STYLE_GUIDE.md:28-29 and the skill allow only YYYY_MM_DD_WORK_NAME folders with SPEC/PLAN/REPORT. Note the global CLAUDE.md tree does name docs/agent/specs|plans|reports, so the two yardsticks disagree; the audit states that in its facts and offers both resolutions.

#### D8. [medium] docs/archive/superpowers/ keeps a tool-named folder with lower-kebab files

Evidence: D:/Projects/sushiai/docs/archive/superpowers/plans/ (3 files, 2 302 to 3 035 lines each) and specs/ (2 files), all named 2026-08-07-*.md. The lifecycle says a work folder moves to archive 'keeping its path below docs/', which would be docs/archive/agent/<YYYY_MM_DD>_<WORK_NAME>/. An untracked .superpowers/sdd/ folder also sits at the repository root.

Recommendation: Archive is frozen, so leave content untouched; ask the owner whether to re-home the five files under docs/archive/agent/ work folders and whether .superpowers/ may be deleted.

Review: confirmed. docs/archive/superpowers/plans (3 files: 2824, 2302, 3035 lines) and specs (2 files: 446, 651 lines), total 9 258 lines, all 2026-08-07-*.md. .superpowers/sdd/2026-08-07-cv-ir-autograd-plan exists at the root, untracked. Severity is closer to low: the archive is frozen and the skill only fixes the shape of what moves in from now on.

#### D9. [medium] Agent work and archive are not reachable from docs/README.md

Evidence: D:/Projects/sushiai/docs/README.md links no file under docs/agent/ or docs/archive/ (13 documents unreachable by link walk). tools/README.md and examples/heavy_mlp/README.md are also unreachable from the index. README.md 'Project layout' docs/ block omits agent/ and archive/.

Recommendation: Add an 'Agent work' and an 'Archive' section to docs/README.md, and a module README list.

Review: confirmed. docs/README.md links nothing under docs/agent/ or docs/archive/; 6 + 2 + 5 = 13 tracked documents there. tools/README.md and examples/heavy_mlp/README.md are not linked either. README.md docs/ block starts at :549 and its link list (:562-573) omits agent and archive.

#### D10. [medium] No module READMEs beside the code

Evidence: No README.md in D:/Projects/sushiai/cli/, apps/, tests/, examples/, or any of include/SushiAI/{core,graph,autograd,nn,optim,data,io,config,train}. docs/agent/reports/2026-09-22-help-screen-task-27.md states 'sa has no cli/README.md; its command list is in the root README'. Module facts live instead in the 578-line root README.md ('What exists today', 'Project layout') and in ARCHITECTURE.md.

Recommendation: Add one README per module (at least cli/, tests/, apps/, and each library module) and move the per-module facts out of the front door.

Review: confirmed. find for README.md outside build/ returns only README.md, docs/README.md, docs/design/README.md, examples/heavy_mlp/README.md, tools/README.md. The repository's own checker cannot catch this: tools/documentation/check_docs_layout.py:166-167 looks only under a modules/ folder, which sushiai does not have, so rule_module_readmes returns nothing here. Module boundary choice (which folders count as modules) is an owner decision the recommendation should have flagged.

#### D11. [medium] Root README is a manual, not a front door, and pins stale measured output

Evidence: D:/Projects/sushiai/README.md is 578 lines: full demo transcript, fusion benchmark tables, mixed-precision analysis, a code walkthrough and a complete directory tree. The transcript and the prose under it disagree: the block says 'Graph : 17 IR nodes over 22 values', the text says 'ops per step: 25 is 21 graph nodes plus one fused update per parameter'. The opening paragraph leads with programme codes ('AI-1 has landed') a newcomer cannot read.

Recommendation: Cut the README to what the library is, requirements, install, one run, links; move the measurements to ARCHITECTURE.md or a guide and reconcile the node count.

Review: confirmed. README.md is 578 lines. :265 'Graph : 17 IR nodes over 22 values' against :308 '25 is 21 graph nodes plus one fused update per parameter'. :10 opens with 'AI-1 has landed'.

#### D12. [medium] CONTRIBUTING.md contradicts the changelog rule and the source-comment rule

Evidence: D:/Projects/sushiai/docs/CONTRIBUTING.md §5: 'Changelog entries start at v1.0.0 ... CHANGELOG.md's [Unreleased] section documents prototype-era milestones informally'; the changelog has 64 formal dated entries and docs/DOCUMENTATION_STYLE_GUIDE.md §5 makes the format mandatory. §4 'Documentation': '@param per parameter, @return for the result' on every public API, against the global rule that they appear only when they state a caller rule; it does not mention the @file header or private-member @brief. §9 says the Discord invite is in SECURITY.md.

Recommendation: Rewrite §4 and §5 to cite DOCUMENTATION_STYLE_GUIDE.md and the source-comments rule instead of restating them.

Review: confirmed. CONTRIBUTING.md:176-178 says changelog entries start at v1.0.0 and the Unreleased section is informal; the file has 62 dated formal entries. :152-153 requires @param and @return on every public API. :248 points to SECURITY.md for the Discord invite, which SECURITY.md:14-15 does carry, so that last clause is an odd pointer and not a broken one.

#### D13. [medium] Stale document paths cited across pages

Evidence: D:/Projects/sushiai/examples/heavy_mlp/README.md:101 live link '../../docs/CLI_GUIDE.md' does not resolve (file is docs/guides/CLI_GUIDE.md). docs/reference/API_REFERENCE.md:12-13 cites `docs/ARCHITECTURE.md` and `docs/INTEGRATION.md` (now docs/architecture/, docs/guides/) and describes docs/README.md as 'what the library is and how to run the demo', which is the root README. .github/workflows/ci.yml:26 cites 'docs/CLAUDE.md' (CLAUDE.md is at root). docs/design/ROADMAP.md:295 and :415 cite `docs/CLI_GUIDE.md`, `docs/ARCHITECTURE.md`. tools/README.md:12 names a module `source/` folder; this repository uses src/.

Recommendation: Fix each path; API_REFERENCE.md is the Doxygen main page (Doxyfile:25,29), so its paths show on the generated site.

Review: confirmed. examples/heavy_mlp/README.md:101 links ../../docs/CLI_GUIDE.md (absent). API_REFERENCE.md:11-13 cites docs/README.md as the demo page, docs/ARCHITECTURE.md and docs/INTEGRATION.md. .github/workflows/ci.yml:26 cites docs/CLAUDE.md. ROADMAP.md:295 and :415 cite the old flat paths. tools/README.md:12 says 'source/'.

#### D14. [medium] ROADMAP.md is a two-repository programme plan with per-section statuses that its single status line hides

Evidence: D:/Projects/sushiai/docs/design/ROADMAP.md: title 'SushiBLAS & SushiAI — SushiStack Construction and AAA-Quality Plan'; sections 5-15 (ST-*, SB-*) are about sushistack and sushiblas; sections 18, 19, 20, 23 carry their own 'Status (2026-08-03): landed' lines while line 3 says 'Open — phase 2 of 4' with no four phases defined in the document (it has three tracks and seven AI-* steps). Line 25 cites `docs/slop/cross_platform_engineering_plan.md` in another repository.

Recommendation: Decide with the owner which repository owns the programme document; keep in sushiai only the AI-* track, move landed sections to archive, and make the status line count phases the document names.

Review: confirmed. ROADMAP.md:1 title covers SushiBLAS and SushiAI; :258-423 are ST-* and SB-* sections; :305, :447, :487, :512, :566, :640 carry their own status lines; :3 says 'phase 2 of 4' and the document defines three tracks and AI-0..AI-7, no four phases; :25 cites docs/slop/cross_platform_engineering_plan.md in the engine repository. Ownership of the programme document is correctly left to the owner.

#### D15. [medium] REMAINING_WORK.md names a class that is not in the code

Evidence: D:/Projects/sushiai/docs/design/REMAINING_WORK.md:7 says '`MemoryPlanner` packs graph IR values' and 'shared between `MemoryPlanner` and `Lowering`'. `grep -rl MemoryPlanner include src` returns 0 files; the header is include/SushiAI/graph/memory_plan.hpp. The audit's F5 checked the paths in that file and not the symbols.

Recommendation: Correct the symbol from memory_plan.hpp in the same edit that fixes the F5 paths, and check the other backticked symbols in the file against the headers (TensorPool, execute_zip_inplace, MixedPrecisionPolicy, StepSpec and num_classes do resolve).

Review: found by review.

#### D16. [medium] The Doxygen path drift reaches code and .gitignore, so a docs-only fix leaves it half done

Evidence: D:/Projects/sushiai/cli/sushiai/services/project.py:558-560 falls back to root / 'docs' / 'Doxyfile', a location the file left. D:/Projects/sushiai/.gitignore:23-25 ignores /docs/api-site/, docs/api-site/html/ and docs/api-site/doxygen-warnings.log, which the Doxyfile (OUTPUT_DIRECTORY = build/docs/api-site) no longer writes. F3 lists only the comment at .gitignore:22.

Recommendation: Fix F3 as one change with one owner of the fact: the Doxyfile holds the output path, the pages and CLAUDE.md cite it, the dead .gitignore rules and the docs/Doxyfile fallback are removed in the same commit (the fallback removal is a code change for the CLI dimension to confirm).

Review: found by review.

#### D17. [medium] The layout checker cannot see missing module READMEs in this repository

Evidence: D:/Projects/sushiai/tools/documentation/check_docs_layout.py:164-172: rule_module_readmes returns at once unless a top-level modules/ folder exists; sushiai keeps its modules under include/SushiAI/<module>/ and src/<module>/. docs/DOCUMENTATION_STYLE_GUIDE.md §8 tells contributors that this checker verifies the layout, so F10 passes the documented gate silently.

Recommendation: Ask the owner which folders are modules in sushiai (the nine library modules, cli/, tests/, apps/), then make the checker read that list from one place rather than hardcoding modules/. Recorded here as a documentation gap; the checker change belongs to the tools dimension.

Review: found by review.

#### D18. [medium] Two yardsticks disagree on the docs/agent shape and the status line position, and the repository follows a third wording

Evidence: Global CLAUDE.md tree: 'docs/agent/specs|plans|reports'. documentation skill: 'docs/agent/<YYYY_MM_DD>_<WORK_NAME>/' with SPEC.md, PLAN.md, REPORT.md and 'No other folder'. sushiai has both shapes (docs/agent/2026_09_16_*/ and docs/agent/reports/). The skill says a design document's status is its 'second line'; D:/Projects/sushiai/docs/DOCUMENTATION_STYLE_GUIDE.md §4 says 'line 3', and ROADMAP.md carries it on line 3. The audit reports the first conflict as a fact and F7, but does not raise it as the decision that blocks F7, F8 and F9.

Recommendation: Put the choice to the owner once, stack-wide, before any file moves: which docs/agent shape is the rule and where the status line sits. Then one of CLAUDE.md, the skill or the style guide changes, and the per-repository findings follow from it.

Review: found by review.

#### D19. [low] Changelog entries break the format in three places

Evidence: D:/Projects/sushiai/docs/reference/CHANGELOG.md:8 'Described this module in its own sushi-module.toml, so hub knows it from the checkout' states a reason; :13 is a blank line splitting the Unreleased list; :3 carries an intro sentence the format does not have. Entries from :14 down cite bare names that do not resolve as written (`rules/norm.cpp`, `services/project.py`, `docs/Doxyfile`, `mainpage.md`), the last two no longer existing.

Recommendation: Drop the reason clause and the blank line; the old cited names are history and can stay if the checker accepts them.

Review: confirmed. The three defects exist but every line number is off by two: the reason clause ('so hub knows it from the checkout') is at CHANGELOG.md:10, the blank line splitting the list is :14, the intro sentence is :3. Entry :9 also joins two changes with 'and'.

#### D20. [low] Humanizer register: manual pages lean on em dashes, bold emphasis and self-praise

Evidence: Em dash counts: README.md 53 in 4 022 words, docs/guides/CLI_GUIDE.md 40 in 3 533, docs/architecture/ARCHITECTURE.md 95 in 8 033, docs/design/ROADMAP.md 118 in 7 410 (the skill's ceiling is about one per 300 words). Phrases: README 'a real, flag-driven training program', 'A genuine D3-resolved', 'an honest list', 'not a wrapper around the example'; ARCHITECTURE.md:869 'Named honestly, because the gap between what a library claims...'; section titles built on corrective contrast ('Fusion is a selector over a catalog, not a code generator', 'structure in the IR, not a runtime switch'). CLAUDE.md ends with 'love you boss, kolay gelsin.' in a project artifact. docs/README.md, GLOSSARY.md and KNOWN_ISSUES.md read in the opposite, generic register ('utilized', 'sub-optimal', triples).

Recommendation: Run a humanizer pass on README.md, CLI_GUIDE.md and ARCHITECTURE.md §5-§7; GLOSSARY.md and KNOWN_ISSUES.md need the rewrite of F1 and F2 anyway.

Review: confirmed. Recounted em dashes and words: README.md 53/4022, CLI_GUIDE.md 40/3533, ARCHITECTURE.md 95/8033, ROADMAP.md 118/7410, all matching. Phrases found at README.md:19, :105, :110, :353, :390, :427 and ARCHITECTURE.md:869. CLAUDE.md ends with 'love you boss, kolay gelsin.'. KNOWN_ISSUES.md:20 'sub-optimal', GLOSSARY.md 'utilized'.

#### D21. [low] Doxyfile and SECURITY links carry stale identifiers

Evidence: D:/Projects/sushiai/Doxyfile:14 'PROJECT_NUMBER = "v1 (unreleased)"' while CMakeLists.txt:16 and cli/pyproject.toml:7 say 0.1.0. docs/reference/SECURITY.md:55,57 link to SushiBLAS and SushiRuntime 'blob/main/docs/SECURITY.md'; in this repository's own layout that file is under docs/reference/, so the sibling URLs are likely stale too (not verified against the sibling repositories).

Recommendation: Set PROJECT_NUMBER from the project version; verify the sibling URLs when those repositories are audited.

Review: unclear. Doxyfile:14 PROJECT_NUMBER = "v1 (unreleased)" against 0.1.0 in CMakeLists.txt:16, cli/pyproject.toml:7 and include/SushiAI/version.hpp:33-37 is confirmed. The SECURITY.md:55,57 sibling URLs are a guess the audit itself marks unverified; I did not open the sibling repositories either, so that half stays open.

### Not checked

- docs/getting_started/TUTORIAL.md (506 lines): only its commands and cited paths were scanned; its numeric claims (ops per step, loss values) were not compared with the code.
- docs/architecture/ARCHITECTURE.md sections 1-6 (lines 1-866): header paths were checked to exist under include/SushiAI/, but class, function and CMake names in the prose were not compared with the source.
- docs/guides/CLI_GUIDE.md lines 67-412 (build, test, run, demo, train/inference flag tables) were not compared with cli/sushiai/cli.py options; no CLI was run.
- docs/guides/INTEGRATION.md body, docs/reference/CODE_OF_CONDUCT.md and SECURITY.md body were not read in full.
- docs/design/ROADMAP.md was scanned by heading and status line only; its 770 lines were not read for accuracy.
- docs/agent/*/SPEC.md, PLAN.md, REPORT.md and the five docs/archive/superpowers files were not read beyond their headers; cited paths inside them were excluded from the path check.
- Anchor fragments were verified only for KNOWN_ISSUES.md to REMAINING_WORK.md and the README to ARCHITECTURE.md §2, §5.1, §5.2, §7 headings by eye; no full anchor check was run.
- External URLs (GitHub sibling repositories, sushisystems.io) were not fetched.
- tools/documentation/check_docs_layout.py and check_changelog.py were not run (no project tools in a read-only task), so their verdicts are not included.
- 'hub install-cli sushiai' in CLI_GUIDE.md and TUTORIAL.md was not verified against the sushistack hub CLI.
- Source-comment and license-header conformance belongs to another dimension and was not measured here.
- Whether changelog entries are in newest-first order within same-day groups was not checked.

## Layout and hygiene

### Facts

- Top-level tracked entries: .clang-format, .editorconfig, .gitignore, .github/ (one workflow), AGENTS.md, CLAUDE.md, CMakeLists.txt, Doxyfile, LICENSE (10777 bytes), README.md (30359 bytes), sushi-module.toml (hub module manifest), apps/ (2 executables + CMakeLists), cli/ (Python CLI package `sushiai`, pyproject, tests), cmake/ (10 files), docs/, examples/ (2 .cpp + heavy_mlp config folder), include/SushiAI/ (52 headers), src/ (41 files), tests/ (50 files), tools/ (7 files).
- Top-level untracked and ignored entries on disk: build/ (223 MB), .claude/ (settings.local.json, 72 bytes), .superpowers/ (2.0 MB, 64 files), cli/.pytest_cache/, cli/sushiai_cli.egg-info/, __pycache__ folders, examples/heavy_mlp/heavy_mlp.ckpt (436 KB).
- 232 files are tracked. By extension: 85 cpp, 55 hpp, 34 md, 24 py, 9 cmake, 6 txt, 5 json, 4 toml, 2 in, 1 yml, 1 h. No binary or image file is tracked.
- Largest tracked files are three archived plans under docs/archive/superpowers/plans/ (141 KB, 126 KB, 112 KB), then src/graph/lowering.cpp (64 KB). Nothing tracked exceeds 150 KB.
- Version declarations: CMakeLists.txt:16 `project(SushiAI VERSION 0.1.0)`; include/SushiAI/version.hpp:40 `SUSHIAI_VERSION_STRING "0.1.0"` (hand-written, guarded by a configure-time mismatch check at CMakeLists.txt:72-79); cli/pyproject.toml:7 `version = "0.1.0"`. These three agree.
- Doxyfile:14 declares `PROJECT_NUMBER = "v1 (unreleased)"`, which does not agree with 0.1.0. sushi-module.toml carries no version field. There is no package.json.
- docs/reference/CHANGELOG.md has a single section heading, `## Unreleased` (line 5), holding 62 entries. No release section exists.
- Git tags: none. Sibling sushiruntime has v0.2.0 and v0.3.0, sushistack has v0.1.0 and v0.2.0, sushiblas has none.
- Remote: origin https://github.com/SushiSystems/SushiAI.git. sushi-module.toml:12 gives the repo as https://github.com/sushisystems/sushiai.git (same repository, different letter case). Module name `sushiai`, alias `sa`.
- Branches: main (ahead of origin/main by 1 commit, e54764b), feature/cv-ir-autograd (tracks origin, fully merged into main), feature/sushicore_help_screen (local only, no upstream, 0 commits ahead of main).
- git status --short prints nothing: no uncommitted and no untracked non-ignored files. No stashes. Last commit 2026-10-04 18:33:46 +0300.
- CI: one workflow, .github/workflows/ci.yml, triggered on push to main and on pull_request. Two jobs: `functional` (ubuntu-latest in intel/oneapi-basekit:latest, direct cmake + ctest) and `cli` (ubuntu-latest, Python 3.11, pyflakes + pytest). No release or tag workflow.
- tools/ holds the four standard checkers plus the shared runner: tools/common/checker.py, tools/documentation/check_source_comments.py (file header, block ceiling, comment runs, separators, history words, @date), check_docs_layout.py (docs entries, names, work folders, status lines, links, module READMEs, archive candidates), check_changelog.py (heading and entry shape, 240 characters, one sentence, nesting), tools/layering/check_layering.py (tiers, upward includes, reaches into another module's source/).
- .gitignore (40 lines) covers: [Bb]uild/, bin/, out/, .vscode/, .vs/, swap files, CMake by-products (CMakeFiles/, CMakeCache.txt, compile_commands.json, _deps/ ...), /docs/api-site/, html/, latex/, *.ckpt, cli/config.local.toml, __pycache__/, *.pyc, cli/*.egg-info/, cli/.pytest_cache/, .sushi_env.json.
- .gitignore has no rule for .claude/ or .superpowers/, yet `git status --ignored` reports both as ignored, so they are excluded by something outside the tracked file (.superpowers/sdd/.gitignore exists; .claude/ is presumably covered by a global or info/exclude rule, not verified).
- No .gitattributes exists, although .editorconfig:5 sets `end_of_line = lf` and the machine is Windows.
- Sibling comparison: sushiruntime and sushiblas use the same include/ + src/ + examples/ + cli/ + cmake/ + root Doxyfile shape as sushiai. sushiblas has no tools/ folder; sushiruntime's tools/ holds only packaging/. sushiai is the only one of the three C++ library repos carrying the four checkers. None of the loose-root clutter found in sushiruntime (CXX-DetectStdlib-*.h, venv/, .scratch_syntax) or sushiengine (about 60 loose images) is present in sushiai.

### Corrections from review

- Extension tally is incomplete: besides the 11 extensions listed, 7 tracked files have other suffixes (py.typed, LICENSE, .gitignore, .editorconfig, Doxyfile, .clang-format, and the second .in is counted). The total of 232 is correct; the listed extensions sum to 226 plus these.
- The audit says the .claude/ ignore source was not verified and that .superpowers/ is ignored 'only by rules outside the tracked file' in a machine-dependent way. Corrected: `git check-ignore -v` shows .claude/settings.local.json is ignored by the global C:/Users/sushi/.config/git/ignore line 1 (`**/.claude/settings.local.json`), and .superpowers/sdd/ by its own .superpowers/sdd/.gitignore line 1 (`*`), which is not machine-dependent.
- F10 cites 'commit 8313fe1-style breaking subjects'. 8313fe1 is a sushistack commit (`refactor(setup)!: ...`), not in sushiai's log.
- F8 cites cli/sushiai/services/project.py lines 558-560 for the docs/Doxyfile fallback; the lines are 557-559.
- F6 implies no repository owns the reference checkers. D:/Projects/sushiskills/tools holds common/, documentation/, layering/ and README.md, so the reference home exists and the tools/README.md text was copied from there.
- Recomputed and correct: 232 tracked files; 62 changelog entries under one `## Unreleased` heading at line 5; version 0.1.0 in three places with Doxyfile at 'v1 (unreleased)'; no tags in sushiai, v0.2.0/v0.3.0 in sushiruntime, v0.1.0/v0.2.0 in sushistack, none in sushiblas; largest tracked files 141175, 126373, 112094 and 64200 bytes.

### Findings

#### L1. [high] CI does not run the four checkers and bypasses the project CLI

Evidence: D:/Projects/sushiai/.github/workflows/ci.yml lines 140-153 call `cmake -B build`, `cmake --build build` and `ctest` directly; lines 184 and 188 call `python -m pyflakes` and `python -m pytest` directly. No step invokes anything under tools/. The file's own header (lines 23-48) records the departure as deliberate. The `continuous-integration` skill requires the four checkers on every push and pull request, and that CI calls the project CLI and nothing else.

Recommendation: Add a checker job that runs the four tools/ scripts (they are standard-library Python and need no toolchain). Decide with the owner whether the cmake/ctest exception stays; if it does, record it in docs/reference/KNOWN_ISSUES.md rather than in a 26-line workflow comment.

Review: confirmed. ci.yml:140-153 calls cmake, cmake --build and ctest; :184 and :188 call pyflakes and pytest; no step references tools/. Recommendation is brick-shaped (one separate checker job).

#### L2. [high] CI covers Linux only and has no release workflow

Evidence: D:/Projects/sushiai/.github/workflows/ci.yml lines 57 and 157: both jobs are `runs-on: ubuntu-latest`. The only trigger block is lines 49-52 (push to main, pull_request). The folder holds one file, ci.yml. The skill requires Windows and Linux on every trigger, a `vX.Y.Z` tag trigger with a packaging build, and one file per trigger named push.yml and release.yml. Sibling sushistack already has ci.yml plus release.yml.

Recommendation: Add a Windows lane (the development machine is Windows, so the primary platform is currently untested in CI), add a release workflow on `v*` tags, and settle the workflow file names across the stack in one decision.

Review: confirmed. ci.yml:57 and :157 are ubuntu-latest; trigger block :49-52 has no tag trigger; .github/workflows holds only ci.yml; sushistack has ci.yml and release.yml.

#### L3. [high] The CI `cli` job installs sushicore from a path that no longer exists

Evidence: D:/Projects/sushiai/.github/workflows/ci.yml:166-179 checks out SushiSystems/SushiStack at main and runs `python -m pip install -e ./.sushistack/sushicore`. SushiStack no longer carries that folder: `git -C D:/Projects/sushistack ls-files | grep -c '^sushicore/'` prints 0, and commit 016c081 `refactor!: move sushicore out of this repository` is contained in origin/main. The install step fails on every run, so the CLI tests never execute in CI. The audit's F4 files this as a stale comment.

Recommendation: Replace the SushiStack checkout and the editable install with one step that installs the CLI package from cli/pyproject.toml, so the sushicore requirement resolves from the declared source. Do it in the same change that removes `|| true` (F3).

Review: found by review.

#### L4. [medium] The pyflakes step can never fail

Evidence: D:/Projects/sushiai/.github/workflows/ci.yml line 184: `run: python -m pyflakes sushiai || true`. The step is named "Import-safety check" but its exit status is discarded. The skill states a job is never marked allowed-to-fail to get green.

Recommendation: Remove `|| true` and fix whatever pyflakes reports, or delete the step.

Review: confirmed. ci.yml:184 reads `run: python -m pyflakes sushiai || true`.

#### L5. [medium] ci.yml comments describe a repository state that no longer exists

Evidence: D:/Projects/sushiai/.github/workflows/ci.yml lines 208-211 say the repo has no Doxyfile and that `Doxyfile` is listed in .gitignore; a root Doxyfile is tracked and .gitignore has no such rule. Line 26 cites `docs/CLAUDE.md`, which does not exist (CLAUDE.md is at the root). Lines 166-179 say sushicore "is published to no index" and install it from a SushiStack checkout, while cli/pyproject.toml:15 depends on `sushicore>=0.7.0` and commit e011cdc is titled "take sushicore from PyPI instead of a sibling checkout". Lines 10-21 state the workflow had never been executed.

Recommendation: Rewrite the workflow header to the current facts, drop the SushiStack checkout if PyPI is now the source, and move the reasoning into a design document.

Review: confirmed. ci.yml:208-211, :26 and :166-179 read as quoted; docs/CLAUDE.md is absent; Doxyfile is tracked. The audit understates it: the sushicore step is broken, not only stale (see missed).

#### L6. [medium] Layering checker is unconfigured, so it enforces nothing

Evidence: D:/Projects/sushiai/tools/layering/check_layering.py line 20: `K_TIER_ORDER: tuple[str, ...] = ()`. Line 79 emits "K_TIER_ORDER is empty; declare the repository's tiers". tools/README.md line 19-20 confirms the checker "does nothing until K_TIER_ORDER lists the repository's tiers". The checker also looks for `modules/<tier>` and `source/` (lines 24, 83), neither of which exists in this single-module `src/` tree.

Recommendation: Declare the tier order for the areas under src/ (core, graph, autograd, nn, optim, train, io, config, data) and adapt the checker's folder constants to this repository's shape, or rename the tree to match the checker (see F7).

Review: confirmed. check_layering.py:20 `K_TIER_ORDER = ()`, :79 emits the empty-order issue, :83 looks under modules/<tier>; tools/README.md:19-20 says the same. src/ has nine area folders.

#### L7. [medium] tools/README.md describes the folder as the stack's reference copy, which it is not

Evidence: D:/Projects/sushiai/tools/README.md lines 3-4: "Reference implementations of the four checkers every Sushi Systems repository carries. A repository copies this folder whole; a rule is fixed here first and copied out." The text was copied in unchanged; sushiai is a consumer. Sibling D:/Projects/sushistack/tools has none of these files (only record_cli_argv.py and tests/), D:/Projects/sushiruntime/tools holds only packaging/, D:/Projects/sushiblas has no tools/.

Recommendation: Name one repository as the owner of the reference checkers, reword this README to say sushiai carries a copy and where the original lives, and roll the same four checkers out to sushiruntime, sushiblas and sushistack.

Review: confirmed. tools/README.md:3-4 reads as quoted; sibling tools/ contents match. The recommendation misses that the owner already exists: D:/Projects/sushiskills/tools holds common/, documentation/, layering/ and README.md.

#### L8. [medium] Folder names depart from the repository-layout yardstick

Evidence: D:/Projects/sushiai/src/ (skill: `source/<area>/<name>.cpp`); D:/Projects/sushiai/apps/ (skill: `applications/`); D:/Projects/sushiai/examples/ is a top-level folder not in the allowed root table; tests are under D:/Projects/sushiai/tests/functional/{unit,integration,regression} plus tests/fixtures/ and tests/package/ (skill: `tests/{unit,integration,regression,common}/` with fixtures under `tests/<kind>/fixtures/`). sushiruntime and sushiblas share the same src/ + examples/ shape, so this is a stack-wide departure rather than a sushiai one.

Recommendation: Take one owner decision for all three library repos: either rename to source/, applications/ and the flat tests/ kinds, or amend the skill to bless src/, apps/ and examples/. Do not rename sushiai alone.

Review: confirmed. src/, apps/, examples/, tests/functional/{unit,integration,regression}, tests/fixtures/ and tests/package/ all exist. tests/common/ does match the skill, which the audit does not say. Recommending one stack-wide decision is the right shape.

#### L9. [medium] Doxyfile sits in the root, where the yardstick forbids it

Evidence: D:/Projects/sushiai/Doxyfile is tracked at the root; the skill lists `Doxyfile` under "Forbidden in the root". D:/Projects/sushiai/.gitignore line 22 says "Doxyfile is committed at docs/Doxyfile", which is false. cli/sushiai/services/project.py lines 558-560 already fall back to docs/Doxyfile, so the move needs no code change. sushiruntime and sushiblas also keep it in the root.

Recommendation: Move Doxyfile to docs/ in all three repos together, which makes the .gitignore comment true.

Review: confirmed. Doxyfile is tracked at root; .gitignore:22 claims docs/Doxyfile; project.py:557-559 falls back to docs/Doxyfile (audit cites 558-560, off by one).

#### L10. [medium] Doxyfile version disagrees with the declared version

Evidence: D:/Projects/sushiai/Doxyfile line 14: `PROJECT_NUMBER = "v1 (unreleased)"`. CMakeLists.txt:16, include/SushiAI/version.hpp:40 and cli/pyproject.toml:7 all say 0.1.0. The generated API site would be stamped v1 for a 0.1.0 library; under the versioning skill v1.0.0 is the owner's decision alone.

Recommendation: Set PROJECT_NUMBER to 0.1.0, or feed it from the CMake project version so that it cannot drift.

Review: confirmed. Doxyfile:14 `PROJECT_NUMBER = "v1 (unreleased)"` against 0.1.0 in CMakeLists.txt:16, version.hpp:40 and cli/pyproject.toml:7.

#### L11. [medium] No release has ever been tagged while 62 entries sit under Unreleased

Evidence: `git tag -l` prints nothing. D:/Projects/sushiai/docs/reference/CHANGELOG.md has one heading, `## Unreleased` at line 5, with 62 `- ` entries. The versioning skill calls for a release proposal at 15 entries, and commit 8313fe1-style breaking subjects aside, the log holds `feat(cli)` commits (e54764b, 1eaf423). Siblings sushiruntime and sushistack are tagged.

Recommendation: Propose v0.1.0 to the owner as the first tag (the declared version is already 0.1.0), cut the changelog section, and bring sushiblas, which is also untagged, along.

Review: confirmed. No tags; CHANGELOG.md has one heading at line 5 and 62 entries. The cited commit 8313fe1 belongs to sushistack, not sushiai; the conclusion does not depend on it.

#### L12. [medium] Two independent version sources with no check between them

Evidence: D:/Projects/sushiai/CMakeLists.txt line 16 (library 0.1.0) and D:/Projects/sushiai/cli/pyproject.toml line 7 (`sushiai-cli` 0.1.0). The CMake guard at lines 72-79 checks only version.hpp. Nothing ties the CLI package version to the library version, and version.hpp is hand-written rather than generated, where the skill says it is generated from `project()`.

Recommendation: State in docs whether the CLI versions with the library or separately; if together, check pyproject against project() in a checker. Consider generating version.hpp with configure_file.

Review: confirmed. CMakeLists.txt:72-79 checks only version.hpp; nothing reads cli/pyproject.toml; version.hpp is hand-written per the comment at :66-71.

#### L13. [medium] One commit on main is unpushed and a local-only branch has no upstream

Evidence: `git branch -vv`: `main e54764b [origin/main: ahead 1] feat(cli): provision this checkout with sa setup, without hub`. `feature/sushicore_help_screen 1eaf423` has no tracking branch; it is an ancestor of main (0 commits ahead), so it holds no unique work. `feature/cv-ir-autograd` is fully merged (0 commits ahead of main).

Recommendation: Push e54764b when the owner approves; it exists only on this machine. Delete the two merged feature branches (local and the remote cv-ir-autograd) with the owner's consent.

Review: confirmed. `git branch -vv` shows main ahead 1 (e54764b), feature/sushicore_help_screen without upstream; both feature branches are 0 commits ahead of main.

#### L14. [medium] The CLI requires a sushicore version that has no tag and may not be published

Evidence: D:/Projects/sushiai/cli/pyproject.toml:15 requires `sushicore>=0.7.0`. D:/Projects/sushicore/pyproject.toml:7 declares 0.7.0, but that repository's tags stop at v0.4.0 and its main is 18 commits ahead of origin/main, and its release workflow publishes on tags. Not verified against PyPI (no network call made). D:/Projects/sushiai/docs/reference/CHANGELOG.md:9 records the floor as 0.4.0 and no entry records the raise to 0.7.0.

Recommendation: Confirm on PyPI which sushicore version is published. Release sushicore 0.7.0 before sushiai's first tag, or lower the floor to the published version, and add the changelog entry for the floor change.

Review: found by review.

#### L15. [medium] docs/guides/CLI_GUIDE.md contradicts the dependency source the repository now uses

Evidence: D:/Projects/sushiai/docs/guides/CLI_GUIDE.md:27 says sushicore 'is published to no index', while cli/pyproject.toml:15 takes it as an index dependency and CHANGELOG.md:11 records 'Took `sushicore` from PyPI instead of a sibling checkout'. The same stale statement sits in ci.yml:167, so the audit's F4 fix would leave the manual page wrong.

Recommendation: Correct the install section of CLI_GUIDE.md in the same change that rewrites the workflow header.

Review: found by review.

#### L16. [low] .gitignore lacks rules for agent and tool leftovers and has a dead docs rule

Evidence: D:/Projects/sushiai/.gitignore has no entry for `.claude/` or `.superpowers/`; they are ignored only by rules outside the tracked file, so a fresh clone on another machine would show them as untracked. There is also no rule for venv/.venv, build-asan/ (`[Bb]uild/` does not match it), imgui.ini or CXX-DetectStdlib-*.h, all of which appear in sibling roots. Lines 23-25 ignore `/docs/api-site/`, but Doxyfile:16 writes to `build/docs/api-site`, so those three lines match nothing.

Recommendation: Adopt one shared .gitignore block across the stack that names .claude/settings.local.json, .superpowers/, venv/, .venv/, build-*/ and the compiler probe headers; remove the dead docs/api-site lines.

Review: confirmed. Dead rule confirmed (.gitignore:23-25 against Doxyfile:16 `build/docs/api-site`); no rule for venv, build-asan or probe headers. The .superpowers claim is wrong in mechanism: it is ignored by its own .superpowers/sdd/.gitignore (`*`), which travels with the folder on any machine.

#### L17. [low] Leftover agent scratch from August sits in the working tree

Evidence: D:/Projects/sushiai/.superpowers/sdd/2026-08-07-cv-ir-autograd-plan/ holds 64 files (2.0 MB), mostly `review-<sha>..<sha>.diff` files plus progress.md and task briefs, for a plan that is already archived at docs/archive/superpowers/plans/2026-08-07-cv-ir-autograd-plan.md and whose branch is merged.

Recommendation: Delete the folder with the owner's consent; the diffs are reproducible from git and the plan is archived.

Review: confirmed. .superpowers holds 64 files, 2.0 MB, review-<sha>..<sha>.diff files plus progress.md; the plan is archived under docs/archive/superpowers/plans/.

#### L18. [low] docs/agent mixes two shapes and docs/archive uses lowercase dated names

Evidence: D:/Projects/sushiai/docs/agent/ holds work folders `2026_09_16_CODE_STYLE_AND_COMMENTS/` and `2026_09_16_DOCS_AND_ROOT_HYGIENE/` (SPEC/PLAN/REPORT, the shape check_docs_layout.py K_WORK_FOLDER expects) beside a flat `reports/` folder with `2026-09-22-help-screen-task-27.md` and `2026-09-22-wave-1-task-7c.md`, lowercase and hyphenated. docs/archive/superpowers/{plans,specs}/ holds five lowercase hyphenated files, against the UPPER_SNAKE_CASE rule for documents.

Recommendation: Pick one docs/agent shape for the whole stack (the global rule says specs|plans|reports, the checker says dated work folders) and file the two September reports accordingly. Archive is frozen, so leave its names and exempt it in the checker if it is not already.

Review: confirmed. docs/agent holds two dated work folders and a flat reports/ with two lowercase hyphenated files; five lowercase files under docs/archive/superpowers. check_docs_layout.py:53-57 already excludes archive, so the suggested exemption exists.

#### L19. [low] No module README beside the code

Evidence: No README.md exists under D:/Projects/sushiai/src/<area>/, D:/Projects/sushiai/include/SushiAI/<area>/, D:/Projects/sushiai/cli/ or D:/Projects/sushiai/tests/ (`ls` of each pattern returned nothing). The nine source areas (autograd, config, core, data, graph, io, nn, optim, train) carry their facts only in the 30 KB root README and docs/architecture/ARCHITECTURE.md.

Recommendation: Decide whether the areas under src/ count as modules. If they do, each gets a README stating ownership, dependencies and entry points; if sushiai is one module, add cli/README.md at least and slim the root README to a front door.

Review: confirmed. No README.md under src/<area>, include/SushiAI/<area>, cli/ or tests/; only examples/heavy_mlp/README.md exists beside code. Root README is 578 lines.

#### L20. [low] Source comments cite a document path that does not exist

Evidence: D:/Projects/sushiai/CMakeLists.txt line 55 and D:/Projects/sushiai/src/CMakeLists.txt line 107 cite `docs/ARCHITECTURE.md`; the file is at docs/architecture/ARCHITECTURE.md. The root CMakeLists.txt also carries multi-line rationale blocks (lines 18-21, 36-47, 50-55) where the skill says the file holds `project()`, options and `add_subdirectory` with no logic, and it embeds the version check at lines 66-80.

Recommendation: Correct both paths, move the version-drift check into a cmake/ function, and move the rationale to the architecture document.

Review: confirmed. CMakeLists.txt:55 and src/CMakeLists.txt:107 cite docs/ARCHITECTURE.md; the file is docs/architecture/ARCHITECTURE.md. Rationale blocks and the version check at :66-80 are as described. Moving the check into a cmake/ function is a brick-shaped fix.

#### L21. [low] No .gitattributes to pin line endings

Evidence: D:/Projects/sushiai/.gitattributes does not exist (`git ls-files .gitattributes` returns 0), while D:/Projects/sushiai/.editorconfig line 5 sets `end_of_line = lf` and the repository is developed on Windows and built on Linux in CI. The skill lists .gitattributes among the root files; sushiruntime has one.

Recommendation: Add `* text=auto eol=lf` with the binary exceptions, matching sushiruntime.

Review: confirmed. No .gitattributes; .editorconfig:5 sets lf; sushiruntime has one (sushiblas and sushistack do not).

#### L22. [low] Manifest URL casing differs from the remote

Evidence: D:/Projects/sushiai/sushi-module.toml line 12: `repo = "https://github.com/sushisystems/sushiai.git"`; `git remote -v` gives `https://github.com/SushiSystems/SushiAI.git`. GitHub resolves both, so nothing breaks, but the two spellings of one fact differ and the product name is written SushiAI everywhere else.

Recommendation: Choose one canonical spelling for repository URLs across every sushi-module.toml and use it in the remote as well.

Review: confirmed. sushi-module.toml:12 lowercase URL against remote SushiSystems/SushiAI.git.

#### L23. [low] A generated checkpoint lies inside the tracked example folder

Evidence: D:/Projects/sushiai/examples/heavy_mlp/heavy_mlp.ckpt (436306 bytes, dated Aug 7) is ignored by .gitignore:32 `*.ckpt`, so it is not tracked, but it sits next to the tracked model.json, train.json and eval.json. The 223 MB build/ tree is likewise ignored and local.

Recommendation: No repository change is needed; delete the stale checkpoint when cleaning the machine. Consider having the example write under build/ so that the source tree stays clean.

Review: confirmed. heavy_mlp.ckpt is 436306 bytes, dated Aug 7, ignored by .gitignore:32. The audit itself says no repository change is needed, so this is a machine-cleanup note, not a repository defect.

### Not checked

- Whether the four checkers pass: no checker, CLI, build or test was run (read-only task), so their current finding counts are unknown.
- The contents of docs/ pages for accuracy, broken links and status lines; only the tree and file names were inspected.
- Source comment headers and license blocks in include/, src/, tests/ and cli/; this belongs to the comments and license dimensions.
- The contents of build/ (223 MB) beyond its top-level listing; skipped as instructed.
- Where the ignore rule for .claude/ comes from (global gitignore or .git/info/exclude); `git check-ignore -v` was not run on it.
- Whether the remote main CI is currently green or has ever run; no network or gh call was made.
- The bodies of cmake/*.cmake, apps/, examples/*.cpp and tests/ for misplaced logic or fixtures; only names and locations were reviewed.
- cli/config.toml and cli/sushistack.deps.toml contents, and whether cli/tests/ placement matches the sibling CLIs.
- The stale-comment claims in ci.yml about test counts (582 tests) and sibling CMake behaviour.
- Full top-level comparison against sushiweb, sushiskills and other siblings; only sushiruntime, sushiblas, sushistack and sushiengine roots were listed.
- The humanizer skill was not invoked before writing this report; the prose was written plainly but not passed through it.

## Code shape

### Facts

- Repository D:/Projects/sushiai, branch main, 232 tracked files: 85 .cpp, 55 .hpp, 1 .h, 24 .py, 34 .md, 10 CMake modules. One static library `sushiai` built from src/CMakeLists.txt; public headers under include/SushiAI/, sources under src/, no modules/<tier>/ tree.
- Library modules (headers files/lines, sources files/lines): core 4/778 + 1/204; graph 12/2766 + 10/4186; autograd 12/947 + 11/1283; nn 7/1241 + 4/398; optim 6/952 + 6/1117; train 2/660 + 2/479; config 4/533 + 3/605; data 1/324 + 1/439; io 2/372 + 1/247. Totals: include 52 files/8725 lines, src 41 files/9205 lines.
- Other trees: tests 50 files/15554 lines (C++ GoogleTest); cli 22 files/1898 lines (Python, Typer, package cli/sushiai); tools 7 files/841 lines (Python checkers); examples 7 files/1196 lines; apps 3 files/425 lines; cmake 10 files/313 lines. No TypeScript, no GLSL.
- Dependency direction measured from #include lines: core -> (SushiBLAS, SushiRuntime); graph -> core; autograd -> graph, core; config -> core (+ nlohmann/json in src/config/value.cpp only); nn -> graph, core, config; optim -> graph (step_recorder only), core, config; data -> core, config; io -> graph, nn, core; train -> graph, autograd, nn, optim, data, config, core. No include cycle found between modules.
- External dependencies: SushiBLAS and SushiRuntime (sibling repositories) and nlohmann/json. nlohmann stays private to src/config/value.cpp. 38 include lines in 27 public headers pull SushiBLAS or SushiRuntime headers.
- Twenty largest tracked source files (lines): src/graph/lowering.cpp 1353; tests/functional/integration/test_autograd_numeric.cpp 1219; src/graph/builder.cpp 908; tests/functional/integration/test_forward.cpp 900; examples/demo_mlp.cpp 898; tests/functional/unit/test_precision.cpp 885; tests/functional/regression/test_fusion_numerics.cpp 820; tests/functional/integration/test_resilience.cpp 806; tests/functional/unit/test_fusion.cpp 681; src/graph/fusion.cpp 664; tests/functional/unit/test_memory_plan.cpp 571; cli/sushiai/services/project.py 562; tests/functional/regression/test_fused_optimizer.cpp 554; include/SushiAI/graph/builder.hpp 541; tests/functional/unit/test_config.cpp 537; tests/functional/regression/test_loss_scaling.cpp 490; tests/functional/unit/test_spatial_shapes.cpp 485; include/SushiAI/graph/ir.hpp 444; src/data/dataset.cpp 439; src/graph/memory_plan.cpp 438.
- Test layout: one binary `sushiai_tests` from tests/functional/{unit (20 files, 234 TEST cases), integration (10 files, 137), regression (10 files, 61)}, shared helpers in tests/common/, fixtures in tests/fixtures/, an installed-package consumer in tests/package/. CTest labels come from the suite-name prefix (tests/CMakeLists.txt:88-102). Three more CTest cases run the demo and the two apps (tests/CMakeLists.txt:119, 155, 166). No tests/golden/ and no tests/benchmark/.
- CLI tests: cli/tests/ holds 7 pytest files (678 lines). tools/ has no tests at all.
- No direct test file exists for: nn/init (kaiming_*, xavier_*, calculate_gain, Nonlinearity are named in no test; only NN::initialize is called as setup), core/allocation (allocate_like named in no test; allocate_arena and view_tensor in one file each), data/dataset (Rings named in one test file, the registry factories make_rings/make_dataset only through test_registries.cpp), optim/step_scalars, graph/fusion_selector (one test file names DeviceFusionSelector), all four tools checkers.
- Error model: the library has one exception type, SushiAI::Error (include/SushiAI/core/common.hpp:113), raised through SA_THROW_IF (common.hpp:130); 335 throw sites in include/ and src/. There are no status enums on fallible operations.
- Logging: the library contains no logger, no log category header and no log call (grep for *_LOG_*, LogCategory, std::cout, printf in include/ and src/ returns nothing). stdout/stderr is written only by apps/ and examples/.
- Comment hygiene measured over 141 C/C++ files: include/ and src/ have no file missing @file or @author, no TODO/FIXME, no separator lines, no stacked // runs and no Doxygen block over 8 lines. All of those defects sit in tests/, examples/ and apps/ (counts in F9).
- Dispatch shapes in the library: autograd uses compile-time tables split per domain (include/SushiAI/autograd/rules/all.hpp:53); layers, optimizers, datasets and objectives use constexpr registry tables looked up by Config::find_entry (include/SushiAI/config/registry.hpp:51); fusion uses a catalog table (src/graph/fusion.cpp:125-147); lowering uses a hand-written if-chain (src/graph/lowering.cpp:391-431).
- Module verdicts: core - needs targeted fixes. graph - needs restructuring (lowering and builder). autograd - sound. nn - needs targeted fixes. optim - needs targeted fixes. train - needs targeted fixes. config - sound. data - needs targeted fixes. io - needs targeted fixes. apps/examples - needs targeted fixes. cli - needs restructuring (services/project.py). tools - needs targeted fixes. tests - needs targeted fixes (comment shape, coverage gaps).

### Corrections from review

- Facts say '10 CMake modules'; git ls-files shows 9 .cmake files. The cmake/ folder holds 10 files because one is a .in template.
- F15 says tools/ has '5 checker scripts, 841 lines'; there are 4 checkers (check_changelog.py, check_docs_layout.py, check_source_comments.py, check_layering.py) plus tools/common/checker.py. The audit's own facts list says four.
- F12 says GraphBuilder has 51 op methods; a count of ValueId-returning declarations in include/SushiAI/graph/builder.hpp gives 45.
- F1 title says a 40-arm chain; lowering.cpp:391-431 has about 33 if statements over 41 op comparisons. 40 is the count of Lowering:: method definitions, which the audit also cites correctly.
- F13 title says seven jobs; the evidence lists eight public functions (build, test, test_package, run, demo, run_app, clean, doxygen).
- F7 says whether the GEMM_BIAS_GELU omission is intended 'was not determined'; tests/functional/unit/test_precision.cpp:241 asserts it is excluded on purpose.
- F15 and the facts say data/dataset is untested beyond Rings in one file; SyntheticClassification is named in 11 test files and the MNIST fallback is tested at tests/functional/integration/test_training.cpp:415-422.
- Recomputed and correct: 232 tracked files; 85 .cpp / 55 .hpp / 24 .py / 34 .md; per-tree file and line counts; twenty largest files; unit 20 files/234 tests, integration 10/137, regression 10/61; 335 throw sites; 0 nodiscard; 38 include lines in 27 headers; 46 files without @file and 49 without @author.

### Findings

#### C1. [high] Lowering dispatches by a 40-arm if-chain that every new operation must extend

Evidence: D:/Projects/sushiai/src/graph/lowering.cpp:391-431 is one `if (node.op == Ops::X.id) { record_x(...); return; }` line per operation, ending in the message at :442-445 'Add it to Lowering::record_node or remove it from the graph'. The class declares 40 Lowering:: methods in one 1353-line file. A new op edits core/ops.hpp, the ALL table (include/SushiAI/core/ops.hpp:244-262), GraphBuilder, Lowering::record_node, lowering.hpp and a rule file. The sibling pass, autograd, solves the same problem with per-domain tables (include/SushiAI/autograd/rules/all.hpp:53-59).

Recommendation: Give lowering the autograd shape: a `LoweringEntry{OpID, record function}` table per domain (elementwise, linalg, activation, loss, shape, spatial, norm) concatenated into one table, plus a static_assert that every op in Ops::ALL that can appear in a graph has an entry. Split lowering.cpp along the same domain files. This is a module-boundary change, so it needs the owner's approval before the wave starts.

Review: confirmed. src/graph/lowering.cpp:391-431 read: one `if (node.op == ...) { record_x(...); return; }` per op, throw at :442-445; 40 Lowering:: definitions counted. '40-arm' is loose: the chain has about 33 if statements covering 41 op comparisons. SOLID check: the table-per-domain fix is brick-shaped and matches autograd; the proposed static_assert must account for BATCHNORM_UPDATE_STATS, which is in Ops::ALL and deliberately has no lowering (tests/functional/integration/test_resilience.cpp:785-797).

#### C2. [high] Lowering helper chains fall through to a default kernel for an unlisted operation

Evidence: D:/Projects/sushiai/src/graph/lowering.cpp:684-686 (`else engine_->elementwise().mul`), :741-745 (`else engine_->nonlinear().tanh(y)`), :756-759 (`else ... tanh_backward`), :612-615 (`else engine_->nonlinear().tanh(c)`), and the `default:` arm at :594-600 that emits GEMM_BIAS for any form not named. An op routed to these helpers by record_node but not added to the inner chain runs the wrong kernel and reports nothing; LEAKY_RELU is in Ops::ACTIVATIONS (ops.hpp:329) and would take the tanh branch at :615 if a catalog entry ever folded it.

Recommendation: Replace each inner chain with a table lookup that throws on a miss, or make the last branch explicit and end with SA_THROW_IF. Covered by the same restructuring as F1.

Review: confirmed. Read lowering.cpp:594-600 (default arm emits GEMM_BIAS), :612-615, :684-686, :741-745, :756-759; every chain ends in an unguarded else. The defect is latent, not live: record_node only routes the listed ops to these helpers today, and the LEAKY_RELU scenario is hypothetical. High is generous; medium fits.

#### C3. [high] TensorPool and Lowering share one header and one source file; four modules include the lowering header only to get the pool

Evidence: D:/Projects/sushiai/include/SushiAI/graph/lowering.hpp:58 (class TensorPool) and :266 (class Lowering); definitions interleaved in src/graph/lowering.cpp:45-202 and :306 onward. include/SushiAI/nn/init.hpp:36, include/SushiAI/io/checkpoint.hpp:41 and include/SushiAI/train/trainer.hpp:48 include graph/lowering.hpp although they use TensorPool, so nn and io depend on the lowering pass and transitively on SushiBLAS engine headers.

Recommendation: Move TensorPool to graph/tensor_pool.hpp and src/graph/tensor_pool.cpp (one primary type per header) and point nn, io and train at it. Private nested key structs at lowering.hpp:209-246 move with it and get their @brief blocks.

Review: confirmed. include/SushiAI/graph/lowering.hpp:58 TensorPool and :266 Lowering; nn/init.hpp:36, io/checkpoint.hpp:41, train/trainer.hpp:48 and graph/compiled_step.hpp:40 include it; init.hpp:174 and checkpoint.hpp:71,113,140 use only Graph::TensorPool. Title says four modules, evidence names three outside graph. Fix is brick-shaped.

#### C4. [high] Exceptions are the public error model; the style rule asks for status values on fallible operations

Evidence: D:/Projects/sushiai/include/SushiAI/core/common.hpp:113-137 defines SushiAI::Error and SA_THROW_IF; 335 throw sites across include/ and src/. File and parse failures throw across the public surface: src/config/value.cpp:324-326 (JSON), :333 and :337 (file read), include/SushiAI/io/checkpoint.hpp:121, 129, 145 (checkpoint open/write). cpp-code-style: an operation that can fail for an outside reason returns an `enum class ...Status`; exceptions never cross a module's public surface. No `[[nodiscard]]` appears anywhere under include/ (grep count 0).

Recommendation: Owner decision, not a mechanical fix: either record in a design document that SushiAI's contract is exception-based and amend the skill for it, or introduce status-returning forms for the file, parse and allocation boundaries (config, io, data, core/allocation). The module wave should not start converting before that is decided.

Review: confirmed. common.hpp:113-137, value.cpp:324-337, checkpoint.hpp:121 and :129 read; 335 throw sites and 0 nodiscard recomputed. Framing it as an owner decision is right.

#### C5. [medium] The library records nothing: no logger, no category, and fallbacks are taken silently

Evidence: grep for `_LOG_`, `LogCategory`, `log_category` in D:/Projects/sushiai/include and src returns no match. Fallbacks that the logging skill lists as must-log happen without a record: the fused-to-unfused GEMM fallback at src/graph/lowering.cpp:544-553 and the device selector choice at :353-359; the MNIST loader returning nullptr when files are absent at src/data/dataset.cpp:242-245; the allocation failure conversion at src/core/allocation.cpp:99-104 and :165-169.

Recommendation: Decide with the owner whether SushiAI uses SushiRuntime's logger or stays silent by contract. If it logs, add one `<module>_log_category.hpp` per module and a WARN at each fallback named here. FusionReport already carries the fusion decisions, so that path may only need a single INFO.

Review: confirmed. No log call in include/ or src/; dataset.cpp:242-245 returns nullptr, allocation.cpp:99-104 and :165 convert, lowering.cpp:544-553 takes the unfused path, :353-359 picks the selector. The allocation sites rethrow as Error, so they are not silent failures, only unlogged.

#### C6. [medium] Third-party types sit in the public interface of every module

Evidence: 38 include lines in 27 headers under D:/Projects/sushiai/include/SushiAI pull SushiBLAS or SushiRuntime. Examples: core/common.hpp:36-37 and :49 (`using DType = SushiBLAS::Core::DataType`), core/ops.hpp:37 and :42 (`using OpID = SushiRuntime::Graph::OpID`), core/allocation.hpp:34, optim/optimizer.hpp:36 (ParameterSlot holds SushiBLAS::Tensor by value at :86-90), train/trainer.hpp:36-38 (SushiRuntime::Core::NodeTiming in RunReport), graph/compiled_step.hpp:34-36 (execution_engine.hpp).

Recommendation: The two libraries are the company's own lower tiers, so this is likely intended, but the `dependencies` skill treats any third-party type in a public header as a decision. Record the decision in docs/design/ and keep the aliasing confined to core/ (DType, OpID, Tensor, span) so the other eight modules include only <SushiAI/core/...>.

Review: confirmed. Recounted: 38 include lines in 27 headers under include/SushiAI pull SushiBLAS or SushiRuntime.

#### C7. [medium] Hand-maintained op lists duplicated and diverging from the fusion catalog

Evidence: D:/Projects/sushiai/src/graph/precision.cpp:55-59 and :98-102 spell the same five-op list twice (MATMUL, GEMM_BIAS, GEMM_BIAS_RELU, GEMM_BIAS_SIGMOID, GEMM_BIAS_TANH) and omit GEMM_BIAS_GELU, while include/SushiAI/graph/fusion.hpp:127-134 `is_gemm_epilogue` includes it and the catalog table at src/graph/fusion.cpp:125-147 is the declared single source. fused_form_name at fusion.hpp:86-97 is a second switch over the same catalog. Whether the GELU omission is intended was not determined.

Recommendation: Derive both precision predicates from the catalog (one `is_matrix_product(OpID)` built on fused_form_of / is_gemm_epilogue) and move the form name into the catalog row. The module wave must confirm against tests/functional/unit/test_precision.cpp:225-234 whether GEMM_BIAS_GELU is excluded on purpose, and state it in one place.

Review: confirmed. precision.cpp:55-59 and :98-102 spell the same five ops; fusion.hpp:86-97 and :127-134 read. The audit left the GELU question open, but the tree answers it: tests/functional/unit/test_precision.cpp:241 asserts narrows_under_mixed_precision(GEMM_BIAS_GELU) is false, with the reason at :224-236. So the recommendation as written (derive from is_gemm_epilogue) would break that test; the catalog row needs its own 'narrows' column instead.

#### C8. [medium] Registry siblings are shaped differently and repeat the same helpers

Evidence: Layers and optimizers have their own registry header (D:/Projects/sushiai/include/SushiAI/nn/layer_registry.hpp:99, include/SushiAI/optim/optimizer_registry.hpp:100); datasets and objectives keep the registry inside the type's main header (include/SushiAI/data/dataset.hpp:282-308, include/SushiAI/train/objective.hpp:241). The `as_span` helper is defined twice (src/nn/layer_registry.cpp:48-52, src/optim/optimizer_registry.cpp:54-58), and LayerEntry / OptimizerEntry / DatasetEntry / ObjectiveEntry are four copies of {type, make}. The layer table lists only linear and gelu (layer_registry.hpp:99-102) while layers.hpp declares ReLU, Tanh and Sigmoid (:90, :122, :138); the dataset table lists only rings (dataset.hpp:308-310) while SyntheticClassification and the MNIST loader exist.

Recommendation: Add data/dataset_registry.hpp and train/objective_registry.hpp so the four registries are shaped alike; move `as_span` and a generic `RegistryEntry<Factory>` into config/registry.hpp. Ask the owner whether the unregistered layers and datasets are meant to be unreachable from a JSON file; if not, add their rows.

Review: confirmed. layer_registry.hpp:99-102 (linear, gelu), optimizer_registry.hpp:100, dataset.hpp:282-310 (rings only), objective.hpp:241; as_span duplicated at src/nn/layer_registry.cpp:48-52 and src/optim/optimizer_registry.cpp:54-58; layers.hpp declares ReLU :90, Tanh :122, Sigmoid :138. Fix is brick-shaped.

#### C9. [medium] tests/, examples/ and apps/ break the source-comment rules that include/ and src/ keep

Evidence: 46 files lack `@file` and 49 lack `@author`: all 40 files under D:/Projects/sushiai/tests/functional, tests/common/test_main.cpp, tests/package/main.cpp, apps/train.cpp, apps/inference.cpp, examples/demo_mlp.cpp, examples/example.cpp (apps/train.cpp:24-42 opens with a 19-line // essay instead). 491 runs of stacked // lines in 45 files, 464 of them under tests/functional (e.g. test_autograd_numeric.cpp 34 runs, test_precision.cpp 33). Separator lines in 16 files, e.g. tests/functional/integration/test_arena.cpp:64-66, tests/functional/unit/test_memory_plan.cpp (12), examples/demo_mlp.cpp (5). 38 Doxygen blocks over 8 lines in 16 files (examples/demo_mlp.cpp:74 is 15 lines; tests/common/test_throws.hpp:26 is 19). History wording at tests/functional/regression/test_fusion_numerics.cpp:584 and tests/functional/unit/test_fusion.cpp:22.

Recommendation: Run a mechanical pass over the three trees: add the file block, move the explanatory paragraphs to docs/ (test rationale belongs in a design or testing guide the test cites), delete separators by splitting files where they mark sections. Check that tools/documentation/check_source_comments.py is run over tests/, examples/ and apps/ in CI; its skip list at :23 does not exclude them, so either it is not run on those paths or it is not gating.

Review: confirmed. Recounted: 46 files without @file (tests 42, apps 2, examples 2), 49 without @author, none in include/ or src/. apps/train.cpp:24-42 and test_arena.cpp:64-66 read. The open question in the recommendation is answerable: .github/workflows/ci.yml runs no tools/ checker at all. Stacked-run and long-block counts were not recomputed.

#### C10. [medium] Library comments carry design reasons and tags outside the allowed set

Evidence: Reason-for-design paragraphs inside Doxygen blocks: D:/Projects/sushiai/include/SushiAI/core/ops.hpp:72-79 ('constexpr rather than consteval because...'), :229-236; include/SushiAI/core/common.hpp:41-47 and :104-110; include/SushiAI/optim/optimizer.hpp:50-68 and :72-78; include/SushiAI/graph/fusion.hpp:47-52; src/nn/layer_registry.cpp:54-60. `@struct`, `@class`, `@enum` and `@throws` tags are used throughout (ops.hpp:49, :296, :308; nn/ilayer.hpp:44; src/core/allocation.cpp:44). `@param` lines restate the name rather than a caller rule (common.hpp:58, :78). Private members without @brief: src/autograd/differentiator.hpp:86-95 (nine members and the constructor at :50), include/SushiAI/graph/lowering.hpp:209-255 (four nested structs, nine members).

Recommendation: Move each reason to docs/design/ and cite the path from a one-sentence @brief; drop @struct/@class/@enum; keep @param only where it states a rule; add `/// @brief` to private members. A full per-symbol count was not taken; the module wave should run the checker per module to get it.

Review: confirmed. ops.hpp:72-79 reason paragraph, differentiator.hpp:86-95 nine bare members, lowering.hpp:209-255 four undocumented nested structs and bare members all read. Sampled only, as the audit says.

#### C11. [medium] Naming and class-shape rules from cpp-code-style are not followed anywhere in the library

Evidence: Constants are bare UPPER_SNAKE, none carries K_: 98 constexpr declarations in include/ and src/ versus 0 uses of `K_` (D:/Projects/sushiai/include/SushiAI/core/common.hpp:55 MAX_RANK, core/ops.hpp:98-262, nn/layer_registry.hpp:99 LAYER_REGISTRY). Enumerators are UPPER_SNAKE instead of PascalCase (graph/fusion.hpp:57-77, core/ops.hpp:302-304). Five interfaces carry the forbidden `I` prefix (graph/fusion_selector.hpp:94, graph/precision.hpp:79, graph/step_recorder.hpp:44, nn/ilayer.hpp:47, train/objective.hpp:119) while the sibling bases Optimizer (optim/optimizer.hpp:95) and Dataset (data/dataset.hpp:75) do not. Two enums have no sized underlying type (optim/optimizer.hpp:49, train/trainer.hpp:230). Single-line `if (...) { ...; return; }` bodies at src/graph/lowering.cpp:391-431 and `if (...) return x;` on the if line (include/SushiAI/config/registry.hpp:55, src/train/trainer.cpp:154).

Recommendation: These are public-surface renames, so they fall under api-stability and need the owner's approval and a version plan. Decide once for the whole stack (SushiBLAS and SushiRuntime appear to use the same UPPER_SNAKE form, judging by DType::FLOAT32 and MAX_TENSOR_RANK), then apply in one rename wave, not module by module.

Review: confirmed. 0 K_ names in include/ and src/; five I-prefixed classes and two unsized enums recomputed at the cited lines; fusion.hpp:57-60 enumerators are UPPER_SNAKE; registry.hpp:55 has the return on the if line. The 98 count was not recomputed.

#### C12. [medium] Several headers hold more than one primary type or responsibility

Evidence: D:/Projects/sushiai/include/SushiAI/nn/layers.hpp declares Linear, ReLU, GELU, Tanh, Sigmoid and Sequential (:50, :90, :106, :122, :138, :156). include/SushiAI/data/dataset.hpp declares Batch, Dataset, SyntheticClassification, Rings, the MNIST loader and the registry (:52, :75, :142, :200, :228-265, :282-308), all implemented in one 439-line src/data/dataset.cpp. include/SushiAI/train/trainer.hpp declares StepSpec, ParameterBinding, TrainingGraph, OptimizerPlacement, TrainerOptions, StepMetrics, RunReport and Trainer (:63-334); RunReport::to_string formats a text report inside src/train/trainer.cpp:90-119. include/SushiAI/nn/ilayer.hpp holds the interface and the LayerAdapter implementation (:47, :82). include/SushiAI/graph/builder.hpp has one GraphBuilder with 51 op methods (class at :114) backed by a 908-line source.

Recommendation: Split per type: one header per layer and per dataset, run_report.hpp out of trainer.hpp, layer_adapter.hpp out of ilayer.hpp. For GraphBuilder ask the owner whether the per-domain split used by autograd (elementwise, linalg, spatial, norm) should apply to the builder as free functions over Graph.

Review: confirmed. Struct and class lines in layers.hpp, dataset.hpp, trainer.hpp and ilayer.hpp match. '51 op methods' on GraphBuilder was not reproduced: a grep for ValueId-returning methods in builder.hpp gives 45; the conclusion does not change.

#### C13. [medium] cli/sushiai/services/project.py is one 562-line module with seven jobs, and it writes to the terminal

Evidence: D:/Projects/sushiai/cli/sushiai/services/project.py defines build (:190), test (:239), test_package (:352), run (:457), demo (:478), run_app (:514), clean (:547), doxygen (:553) plus DLL deployment (:279) and toolchain argument assembly (:117, :165). It calls console.header/info/error/success throughout (:192, :204-206, :228, :232-234) and console.console.print at :474, :510, :543; python-code-style: a service never writes to the terminal. Module-level singletons `_RUNNER` (:90) and `_DRIVER`. 8 of 16 symbols have no docstring (BuildType :38, Suite :44, _build_dir :86, build :190 among them). Enum members are lowercase (:39-41, :47-56); constants lack the K_ prefix (:68 _SUITE_LABEL_REGEX, :79 _DEMO_TARGETS, :83). The module docstring is 16 lines against a ceiling of 6 (:1-16); stacked # comments at :45-46, :52-54, :59-63, :75-78. cli.py has separator lines at :38-40.

Recommendation: Split into services/build.py, test.py, package_test.py, run.py with a small result type returned to the command layer, which does the printing. Pass the driver and runner in instead of holding them at module level. Compare with the sibling CLIs before reshaping so all `s*` CLIs come out alike; that comparison belongs to the CLI dimension.

Review: confirmed. project.py: eight public functions at the cited lines (the title says seven jobs), _RUNNER :90 and _DRIVER :91, console calls throughout, 16-line module docstring at :1-16. The audit does not say that cli/sushiai has no commands/ package at all; see missed.

#### C14. [medium] The layering checker cannot check this repository

Evidence: D:/Projects/sushiai/tools/layering/check_layering.py:20 sets `K_TIER_ORDER: tuple[str, ...] = ()` and :58-62 `collect` only returns a tree when `<root>/modules` is a directory. The repository has no modules/ folder (sources are include/SushiAI/<module> and src/<module>), so the checker collects nothing and reports no issue. The dependency order in the facts above is therefore unenforced.

Recommendation: Either declare the order (core < graph < autograd < config < nn/optim/data < io < train) and teach the checker the include/src layout, or move the tree to modules/<tier>/<module>/ as repository-layout describes. The second is a boundary decision for the owner.

Review: confirmed. check_layering.py:20 K_TIER_ORDER is empty and collect() at :58-62 requires <root>/modules; the repository root has no modules/ folder.

#### C15. [medium] Units with no direct test

Evidence: No test names kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform, calculate_gain or Nonlinearity (D:/Projects/sushiai/include/SushiAI/nn/init.hpp:47-111; src/nn/init.cpp has a switch at :39). allocate_like (src/core/allocation.cpp:135) is named in no test. No test file includes or targets optim/step_scalars, graph/fusion_selector (one file names DeviceFusionSelector), or data/dataset beyond Rings in one file; make_sgd, make_adamw and make_rings are reached only through tests/functional/integration/test_registries.cpp. tools/ (5 checker scripts, 841 lines) has no test. There is no tests/unit file for lowering; it is covered only through integration tests.

Recommendation: Add tests/functional/unit/test_init.cpp (gain values, fan computation, fixed-seed distribution bounds with stated tolerance), test_allocation.cpp, test_dataset.cpp, test_step_scalars.cpp; add pytest files for the four checkers with a small fixture tree each.

Review: confirmed. Symbol greps over tests/ reproduce: kaiming, xavier, calculate_gain, Nonlinearity, allocate_like, make_sgd, make_adamw, make_rings 0 files; Rings 1; DeviceFusionSelector 1. Two corrections: SyntheticClassification is named in 11 test files and try_load_mnist is tested at test_training.cpp:415-422, so 'data/dataset beyond Rings' understates coverage; tools/ has four checkers, not five.

#### C16. [medium] CI runs none of the tools/ checkers, and the one lint step cannot fail

Evidence: D:/Projects/sushiai/.github/workflows/ci.yml has no step that calls tools/documentation/check_source_comments.py, check_changelog.py, check_docs_layout.py or tools/layering/check_layering.py (grep for 'check_' and 'tools/' returns nothing). Line 184 runs `python -m pyflakes sushiai || true`, so it never fails. F9 left this as an either/or; the file settles it.

Recommendation: Add one CI step that runs the four checkers through the CLI and gates the job; remove `|| true` from the pyflakes step. Do this after F9's cleanup, or the first run fails on 46 files.

Review: found by review.

#### C17. [medium] The CLI has no commands/ layer; cli.py holds every command

Evidence: D:/Projects/sushiai/cli/sushiai contains cli.py (262 lines), config.py, console.py, env.py and services/ only; there is no commands/ package. python-code-style: commands/ parses arguments, services/ does the work. F13 proposes returning results to 'the command layer' without noting that the layer does not exist as a package.

Recommendation: Create cli/sushiai/commands/ with one file per command group (build, test, run, provision) when project.py is split, so the service split and the command split land as one change. Align the shape with the sibling s* CLIs.

Review: found by review.

#### C18. [medium] core/allocation reaches a SushiRuntime singleton instead of taking the strategy as a dependency

Evidence: D:/Projects/sushiai/src/core/allocation.cpp:97 and :163 pass `SushiRuntime::Memory::Strategies::Shared::instance()` inside allocate functions. cpp-code-style class shape rule 9: dependencies arrive through the constructor, never through a global or a singleton. Every tensor the library allocates is tied to that one strategy.

Recommendation: Owner decision on data ownership: either pass the memory strategy in (through the engine context the functions already receive) or record in docs/design/ that Shared is the only strategy SushiAI supports.

Review: found by review.

#### C19. [medium] An operation in the public Ops::ALL table has no builder, lowering or backward rule

Evidence: D:/Projects/sushiai/include/SushiAI/core/ops.hpp:193 declares BATCHNORM_UPDATE_STATS and :261 lists it in ALL; it has no arm in src/graph/lowering.cpp:391-431. tests/functional/integration/test_resilience.cpp:785-797 and tests/functional/unit/test_backward.cpp:75-78 pin that it is rejected and name it 'CV-7'. The audit left it under not_checked as a dead-code candidate.

Recommendation: Confirm CV-7 is listed in docs/design/REMAINING_WORK.md. F1's 'every op in ALL has a lowering entry' static_assert needs an explicit not-lowered list that this op sits in, or the assert cannot compile.

Review: found by review.

#### C20. [low] Swallowing catch blocks and unreachable-return idiom

Evidence: D:/Projects/sushiai/examples/demo_mlp.cpp:243-247, :267-270, :290-293 catch std::logic_error with a comment-only body (the failure is then rethrown as SA_THROW_IF two lines later, so it is not lost, but the block shape is the forbidden one and the three parsers are near-identical copies). `SA_THROW_IF(true, ...)` followed by a dummy return appears at include/SushiAI/config/registry.hpp:66-67, src/config/value.cpp:326-328, src/train/trainer.cpp:208-211, src/graph/lowering.cpp:442.

Recommendation: Fold the three demo parsers into one template that throws from inside the catch. Add an `SA_THROW(...)` macro or a `[[noreturn]]` helper so the dummy returns go away.

Review: confirmed. demo_mlp.cpp:243-247 comment-only catch followed by SA_THROW_IF; registry.hpp:66-67 and value.cpp:326-328 dummy returns read. lowering.cpp:442 has the SA_THROW_IF(true) but no dummy return (void function).

#### C21. [low] Umbrella header and quoted includes

Evidence: D:/Projects/sushiai/include/SushiAI/SushiAI.h:32-93 includes every public header with quoted relative paths (`#include "config/value.hpp"`), against the rule 'always <Project/module/name.hpp>' and the listed mistake 'a header per module that includes everything'. apps/train.cpp:49, apps/inference.cpp, examples/*.cpp and 3 test files include it. src/autograd/differentiator.cpp:34 and src/autograd/tape.cpp:31 use a quoted include for the private differentiator.hpp, which sits in src/ beside the sources.

Recommendation: Keep the umbrella only if the owner wants it as a convenience for applications; switch it to angle includes, and make the library's own sources and tests include exact headers.

Review: confirmed. SushiAI.h:32-39 uses quoted relative includes; included by apps/train.cpp, apps/inference.cpp, both examples, tests/common/test_utils.hpp, test_engine_smoke.cpp and tests/package/main.cpp. Because test_utils.hpp pulls the umbrella, most tests see every header, which weakens F15's include-based coverage reading.

#### C22. [low] Tests and apps run from CMake/CTest, which doctrine item 7 forbids

Evidence: D:/Projects/sushiai/tests/CMakeLists.txt:88-102 (gtest_discover_tests), :119, :155, :166 (add_test running sushiai_demo_mlp and the two apps); cli/sushiai/services/project.py:273-276 runs them through `_DRIVER.ctest_run`. sushiskills doctrine 7: never run a program or a test from CMake (ctest, a custom target, a post-build step).

Recommendation: This is the shared sushicore driver's design, so it is a stack-level question for the owner: either the CLI launches the test binary itself with its own label filter, or the doctrine sentence is amended to allow ctest when the CLI drives it.

Review: confirmed. tests/CMakeLists.txt:88-102 gtest_discover_tests and add_test at :119, :155, :166 read. It breaks doctrine 7 as written; treating it as a stack-level owner question is correct.

### Not checked

- Line-by-line reading of the library: only these were read in part or whole - core/ops.hpp, core/common.hpp, src/core/allocation.cpp, config/registry.hpp, src/config/value.cpp (lines 300-340), nn/ilayer.hpp, nn/layer_registry.hpp, src/nn/layer_registry.cpp, optim/optimizer.hpp (44-130), src/optim/optimizer_registry.cpp (36-120), autograd/rule.hpp, autograd/rules/all.hpp, src/autograd/differentiator.hpp, graph/fusion.hpp (44-165), graph/lowering.hpp (196-262), src/graph/lowering.cpp (outline, 374-446, 536-622, 676-765), src/graph/precision.cpp (40-112), src/train/trainer.cpp (36-262), data/dataset.hpp (270-324), apps/train.cpp, examples/demo_mlp.cpp (236-296), cli/sushiai/services/project.py (1-90, 186-280), cli/sushiai/cli.py (1-60), tools/layering/check_layering.py, tools/documentation/check_source_comments.py (1-60).
- Not opened at all: src/graph/builder.cpp, fusion.cpp body, memory_plan.cpp, cast_insertion.cpp, rematerialise.cpp, arena.cpp, compiled_step.cpp, fusion_selector.cpp; all seven src/autograd/rules/*.cpp, tape.cpp, gradcheck.cpp; src/nn/layers.cpp, layer_stack.cpp, init.cpp; src/optim/adamw.cpp, sgd.cpp, loss_scaler.cpp, step_scalars.cpp, optimizer.cpp; src/train/objective.cpp; src/data/dataset.cpp body; src/io/checkpoint_format.cpp; src/config/model_config.cpp, train_config.cpp; core/shape.hpp; nn/module.hpp; apps/inference.cpp; examples/example.cpp.
- Per-symbol @brief coverage for the whole library: only sampled (F10). The repository's checker was not run, because the task forbids running project tooling.
- Dead code: no systematic search for unused functions or unreachable branches was made. UpdateKernel::SEPARATE (optim/optimizer.hpp:60-68) and BATCHNORM_UPDATE_STATS (core/ops.hpp:193, absent from the lowering chain) are candidates the module wave should check.
- Test content: determinism (seeds, float tolerances), test names, and whether each regression test maps to a named defect were not reviewed; only counts, layout and symbol references were measured. No wall-clock or random_device use was found by grep, which is the extent of the check.
- Hot-loop rules (allocation or virtual dispatch per element), include-order grouping, const/noexcept coverage and rule-of-five declarations were not reviewed.
- Whether the GEMM_BIAS_GELU omission in src/graph/precision.cpp is intended (F7).
- cli/sushiai/config.py, console.py, env.py, services/diag.py, services/discovery.py and all cli/tests/ files were only measured (docstring presence, first line), not read. tools/common/checker.py, check_changelog.py and check_docs_layout.py were not read.
- CMake files under cmake/ and the three CMakeLists.txt were not reviewed as code; build configuration is outside this dimension apart from the test wiring cited.
- The licence header text itself and the repository layout against repository-layout (include/src versus source/, tests/functional versus tests/unit) belong to other dimensions and were not assessed here.
- Coverage facts in F15 come from grepping test files for symbol names and header includes; a unit exercised only through the umbrella header or indirectly through Trainer would still show as untested by name.
