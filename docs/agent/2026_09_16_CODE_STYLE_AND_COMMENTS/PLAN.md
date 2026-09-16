# Kaynak Kod Yorumları ve Kod Stili Standartlaştırma Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate all 631 source comment rule violations across `include/` and `src/` to achieve full compliance with `source-comments` and `cpp-code-style` skills.

**Architecture:** Systematic, module-by-module cleanup of file headers, symbol comment blocks, multi-line comment runs, separators, and forbidden tags. Verification is performed exclusively through the static Python checker `tools/documentation/check_source_comments.py`.

**Tech Stack:** Python 3.11 (standard library only), C++17, Doxygen comment standards, Git.

**Spec:** [`docs/agent/2026_09_16_CODE_STYLE_AND_COMMENTS/SPEC.md`](SPEC.md)

## Global Constraints

- Never invoke `sa build`, `ninja`, `cmake`, or compiler binaries. All verifications must be static checks via Python scripts.
- Every source file begins with the standard Apache-2.0 license block, followed by exactly one blank line, followed by the `@file` Doxygen block (at most 6 lines, including `@file`, `@brief`, `@author`).
- In header files, `#pragma once` is placed immediately after the `@file` block.
- Symbol doc blocks (`/** ... */` or consecutive `///`) must not exceed 8 lines (`rule_block_ceiling`).
- Inside function bodies, multi-line `//` comment runs are forbidden. A `//` comment must be a single line stating a concrete invariant (`rule_comment_runs`).
- Separator lines (`// ---------`, `// ========`) are forbidden outside license blocks (`rule_separators`).
- `TODO`, `FIXME`, and history markers are forbidden (`rule_history`).

---

### Task 1: Top-Level Files and IO Module

**Files:**
- Modify: `include/SushiAI/SushiAI.h`
- Modify: `include/SushiAI/version.hpp`
- Modify: `src/version.cpp`
- Modify: `include/SushiAI/io/checkpoint.hpp`
- Modify: `include/SushiAI/io/checkpoint_format.hpp`
- Modify: `src/io/checkpoint_format.cpp`

**Steps:**
- [x] **Step 1: Check baseline violations**
  Run: `python tools/documentation/check_source_comments.py include/SushiAI/SushiAI.h include/SushiAI/version.hpp src/version.cpp include/SushiAI/io src/io`
  Expected: 16 violations reported.
- [x] **Step 2: Fix headers and comments in top-level and io files**
  - Add canonical `@file` blocks after license blocks in all 6 files.
  - In `checkpoint.hpp`, condense blocks at lines 96, 139, 173 to <= 8 lines and simplify multi-line `//` run at line 196 to a single-line invariant.
  - In `src/io/checkpoint_format.cpp`, remove separator lines and simplify `//` runs.
- [x] **Step 3: Verify clean check**
  Run: `python tools/documentation/check_source_comments.py include/SushiAI/SushiAI.h include/SushiAI/version.hpp src/version.cpp include/SushiAI/io src/io`
  Expected: 0 findings, exit code 0.
- [ ] **Step 4: Commit**
  ```bash
  git commit -m "docs(io): align source comments and file headers to standards"
  ```

---

### Task 2: Data and Config Modules

**Files:**
- Modify: `include/SushiAI/data/dataset.hpp`
- Modify: `src/data/dataset.cpp`
- Modify: `include/SushiAI/config/model_config.hpp`
- Modify: `include/SushiAI/config/registry.hpp`
- Modify: `include/SushiAI/config/train_config.hpp`
- Modify: `include/SushiAI/config/value.hpp`
- Modify: `src/config/model_config.cpp`
- Modify: `src/config/train_config.cpp`
- Modify: `src/config/value.cpp`

**Steps:**
- [x] **Step 1: Check baseline violations**
  Run: `python tools/documentation/check_source_comments.py include/SushiAI/data src/data include/SushiAI/config src/config`
  Expected: 50 violations reported.
- [x] **Step 2: Fix headers, ceiling, and separators in data and config**
  - Align `@file` headers to <= 6 lines.
  - Truncate over-ceiling symbol blocks to <= 8 lines.
  - Remove separators and condense multi-line `//` runs into single-line invariants.
- [x] **Step 3: Verify clean check**
  Run: `python tools/documentation/check_source_comments.py include/SushiAI/data src/data include/SushiAI/config src/config`
  Expected: 0 findings, exit code 0.
- [ ] **Step 4: Commit**
  ```bash
  git commit -m "docs(config): align data and config source comments to standards"
  ```

---

### Task 3: Core and Train Modules

**Files:**
- Modify: `include/SushiAI/core/common.hpp`
- Modify: `include/SushiAI/core/shape.hpp`
- Modify: `include/SushiAI/core/tensor.hpp`
- Modify: `include/SushiAI/core/ops.hpp`
- Modify: `src/core/shape.cpp`
- Modify: `src/core/tensor.cpp`
- Modify: `include/SushiAI/train/trainer.hpp`
- Modify: `include/SushiAI/train/objective.hpp`
- Modify: `include/SushiAI/train/step_report.hpp`
- Modify: `src/train/trainer.cpp`
- Modify: `src/train/objective.cpp`

**Steps:**
- [x] **Step 1: Check baseline violations**
  Run: `python tools/documentation/check_source_comments.py include/SushiAI/core src/core include/SushiAI/train src/train`
  Expected: 85 violations reported.
- [x] **Step 2: Clean comments in core and train**
  - Ensure canonical `@file` header in all files.
  - Condense multi-line `//` comments into single-line invariant statements.
  - Shorten Doxygen blocks exceeding 8 lines.
  - Remove all separator comment lines.
- [x] **Step 3: Verify clean check**
  Run: `python tools/documentation/check_source_comments.py include/SushiAI/core src/core include/SushiAI/train src/train`
  Expected: 0 findings, exit code 0.
- [ ] **Step 4: Commit**
  ```bash
  git commit -m "docs(core): align core and train source comments to standards"
  ```

---

### Task 4: NN and Optim Modules

**Files:**
- Modify: `include/SushiAI/nn/module.hpp`
- Modify: `include/SushiAI/nn/parameter.hpp`
- Modify: `include/SushiAI/nn/layer.hpp`
- Modify: `include/SushiAI/nn/linear.hpp`
- Modify: `include/SushiAI/nn/activation.hpp`
- Modify: `include/SushiAI/nn/loss.hpp`
- Modify: `include/SushiAI/nn/conv2d.hpp`
- Modify: `include/SushiAI/nn/layer_stack.hpp`
- Modify: `src/nn/layer.cpp`
- Modify: `src/nn/linear.cpp`
- Modify: `src/nn/activation.cpp`
- Modify: `src/nn/conv2d.cpp`
- Modify: `src/nn/layer_stack.cpp`
- Modify: `include/SushiAI/optim/optimizer.hpp`
- Modify: `include/SushiAI/optim/sgd.hpp`
- Modify: `include/SushiAI/optim/adamw.hpp`
- Modify: `include/SushiAI/optim/loss_scaler.hpp`
- Modify: `src/optim/sgd.cpp`
- Modify: `src/optim/adamw.cpp`
- Modify: `src/optim/loss_scaler.cpp`

**Steps:**
- [x] **Step 1: Check baseline violations**
  Run: `python tools/documentation/check_source_comments.py include/SushiAI/nn src/nn include/SushiAI/optim src/optim`
  Expected: 120 violations reported.
- [x] **Step 2: Clean comments in nn and optim**
  - Fix `@file` blocks and `#pragma once` positioning.
  - Reduce long doc comments to max 8 lines.
  - Eliminate multi-line `//` blocks in kernel/optimization logic.
  - Remove separators and TODO markers.
- [x] **Step 3: Verify clean check**
  Run: `python tools/documentation/check_source_comments.py include/SushiAI/nn src/nn include/SushiAI/optim src/optim`
  Expected: 0 findings, exit code 0.
- [x] **Step 4: Commit**
  ```bash
  git commit -m "docs(nn): align nn and optim source comments to standards"
  ```

---

### Task 5: Autograd Module

**Files:**
- Modify: `include/SushiAI/autograd/backward.hpp`
- Modify: `include/SushiAI/autograd/gradcheck.hpp`
- Modify: `include/SushiAI/autograd/rule.hpp`
- Modify: `include/SushiAI/autograd/tape.hpp`
- Modify: `include/SushiAI/autograd/rules/activation.hpp`
- Modify: `include/SushiAI/autograd/rules/all.hpp`
- Modify: `include/SushiAI/autograd/rules/elementwise.hpp`
- Modify: `include/SushiAI/autograd/rules/linalg.hpp`
- Modify: `include/SushiAI/autograd/rules/loss.hpp`
- Modify: `include/SushiAI/autograd/rules/norm.hpp`
- Modify: `include/SushiAI/autograd/rules/shape.hpp`
- Modify: `include/SushiAI/autograd/rules/spatial.hpp`
- Modify: `src/autograd/differentiator.cpp`
- Modify: `src/autograd/differentiator.hpp`
- Modify: `src/autograd/gradcheck.cpp`
- Modify: `src/autograd/tape.cpp`
- Modify: `src/autograd/rules/activation.cpp`
- Modify: `src/autograd/rules/elementwise.cpp`
- Modify: `src/autograd/rules/linalg.cpp`
- Modify: `src/autograd/rules/loss.cpp`
- Modify: `src/autograd/rules/norm.cpp`
- Modify: `src/autograd/rules/shape.cpp`
- Modify: `src/autograd/rules/spatial.cpp`

**Steps:**
- [x] **Step 1: Check baseline violations**
  Run: `python tools/documentation/check_source_comments.py include/SushiAI/autograd src/autograd`
  Expected: 72 violations reported.
- [x] **Step 2: Clean autograd headers and rule implementations**
  - Standardize `@file` blocks across all rule translation units.
  - Convert multi-line `//` comments into single-line invariants.
  - Remove separators and clip docblocks to 8 lines.
- [x] **Step 3: Verify clean check**
  Run: `python tools/documentation/check_source_comments.py include/SushiAI/autograd src/autograd`
  Expected: 0 findings, exit code 0.
- [x] **Step 4: Commit**
  ```bash
  git commit -m "docs(autograd): align autograd rules and headers to comment standards"
  ```

---

### Task 6: Graph Module

**Files:**
- Modify: `include/SushiAI/graph/arena.hpp`
- Modify: `include/SushiAI/graph/builder.hpp`
- Modify: `include/SushiAI/graph/cast_insertion.hpp`
- Modify: `include/SushiAI/graph/compiled_step.hpp`
- Modify: `include/SushiAI/graph/fusion.hpp`
- Modify: `include/SushiAI/graph/fusion_selector.hpp`
- Modify: `include/SushiAI/graph/ir.hpp`
- Modify: `include/SushiAI/graph/lowering.hpp`
- Modify: `include/SushiAI/graph/memory_plan.hpp`
- Modify: `include/SushiAI/graph/precision.hpp`
- Modify: `include/SushiAI/graph/rematerialise.hpp`
- Modify: `include/SushiAI/graph/step_recorder.hpp`
- Modify: `src/graph/arena.cpp`
- Modify: `src/graph/builder.cpp`
- Modify: `src/graph/cast_insertion.cpp`
- Modify: `src/graph/compiled_step.cpp`
- Modify: `src/graph/fusion.cpp`
- Modify: `src/graph/fusion_selector.cpp`
- Modify: `src/graph/lowering.cpp`
- Modify: `src/graph/memory_plan.cpp`
- Modify: `src/graph/precision.cpp`
- Modify: `src/graph/rematerialise.cpp`

**Steps:**
- [x] **Step 1: Check baseline violations**
  Run: `python tools/documentation/check_source_comments.py include/SushiAI/graph src/graph`
  Expected: 288 violations reported.
- [x] **Step 2: Standardize headers, docblocks, and body comments**
  - Standardize file opening blocks across all IR, lowering, memory planning, and fusion files.
  - Clamp docblocks exceeding 8 lines.
  - Replace runs of `//` lines with single-line invariants.
  - Remove all ASCII separator bars.
  - Relocate any remaining `TODO` or `FIXME` notes to `docs/design/REMAINING_WORK.md`.
- [x] **Step 3: Verify clean check**
  Run: `python tools/documentation/check_source_comments.py include/SushiAI/graph src/graph`
  Expected: 0 findings, exit code 0.
- [x] **Step 4: Commit**
  ```bash
  git commit -m "docs(graph): align graph IR, passes, and lowering to comment standards"
  ```

---

### Task 7: Full Repository Verification and Changelog Entry

**Files:**
- Modify: `docs/reference/CHANGELOG.md`
- Create: `docs/agent/2026_09_16_CODE_STYLE_AND_COMMENTS/REPORT.md`

**Steps:**
- [x] **Step 1: Run full source comments check across include and src**
  Run: `python tools/documentation/check_source_comments.py include src --report`
  Expected: 0 total findings across all rules.
- [x] **Step 2: Run docs layout and changelog checkers**
  Run: `python tools/documentation/check_docs_layout.py . --report`
  Run: `python tools/documentation/check_changelog.py . --report`
  Expected: 0 findings for each.
- [x] **Step 3: Update CHANGELOG.md and create REPORT.md**
  - Add standard changelog entry under `## Unreleased`.
  - Write factual completion report into `REPORT.md`.
- [x] **Step 4: Commit**
  ```bash
  git commit -m "docs(comments): complete source comments standardization across core and src"
  ```
