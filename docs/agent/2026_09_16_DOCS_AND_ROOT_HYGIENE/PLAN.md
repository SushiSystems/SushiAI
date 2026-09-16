# Dokümantasyon Ağacı ve Kök Dizin Standartlaştırılması Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish root directory hygiene according to `repository-layout`, restructure `docs/` into the canonical `documentation` skill tree, update CLI Doxyfile discovery, remove tracked HTML build artifacts, and update all internal links.

**Architecture:** Move root-forbidden files (`Doxyfile`) to `docs/`, create root configuration and agent files (`README.md`, `CLAUDE.md`, `AGENTS.md`, `.clang-format`, `.editorconfig`), reorganize `docs/` into categorized directories (`getting_started/`, `architecture/`, `guides/`, `reference/`, `design/`, `archive/`), update CLI lookup logic, and rewrite internal documentation links to prevent breakage.

**Tech Stack:** Markdown, Python (Typer CLI), CMake, Doxygen, Clang-Format, Git.

**Spec:** [`docs/agent/2026_09_16_DOCS_AND_ROOT_HYGIENE/SPEC.md`](SPEC.md)

## Global Constraints

- Root holds only allowed files: `README.md`, `LICENSE`, `CMakeLists.txt`, `AGENTS.md`, `CLAUDE.md`, `.clang-format`, `.editorconfig`, `.gitignore`, `.gitattributes`, `cmake/`, `cli/`, `tools/`, `modules/`, `tests/`, `applications/`, `third_party/`, `docs/`.
- `docs/` directory contains only allowed directories: `getting_started/`, `architecture/`, `guides/`, `reference/`, `design/`, `agent/`, `archive/` plus root `README.md`, `CONTRIBUTING.md`, `DOCUMENTATION_STYLE_GUIDE.md`.
- No redirection stub in root `README.md`: root `README.md` must be the complete, self-contained project manual for GitHub.
- `CHANGELOG.md` lines follow the strict 240-char max single-sentence past-tense format: `- YYYY-MM-DD — <scope>: <Past-tense verb> <what changed> (<where>).`
- Every link and cited path must resolve; no dead links.
- All builds and test commands must go through `sa` CLI (`sa test`, `sa doxygen`).

---

### Task 1: Root Directory Hygiene & Gitignore

**Files:**
- Create: `README.md`
- Create: `CLAUDE.md`
- Create: `AGENTS.md`
- Create: `.clang-format`
- Create: `.editorconfig`
- Move: `Doxyfile` -> `docs/Doxyfile`
- Delete from git: `docs/CLAUDE.md`
- Modify: `.gitignore`
- Untrack: `docs/api-site/`

**Interfaces:**
- Produces: Root standard files recognized by git, IDEs, and agent harnesses.

- [ ] **Step 1: Copy full documentation content to root `README.md`**
Copy the comprehensive documentation currently in `docs/README.md` to `README.md` at root, adjusting links that point to sibling files in `docs/`.

- [ ] **Step 2: Create root `CLAUDE.md` and `AGENTS.md`**
Move `docs/CLAUDE.md` to root `CLAUDE.md`, adding explicit declarations for the SushiStack skills (`sushiskills`, `repository-layout`, `cpp-code-style`, `documentation`, `project-tools`, `source-comments`, `logging`, `continuous-integration`, `humanizer`). Create root `AGENTS.md` mirroring agent guidance.

- [ ] **Step 3: Create `.clang-format` and `.editorconfig`**
Create `.clang-format` encoding LLVM base, 4-space indent, 100 column limit, Allman braces, access specifier indentation. Create `.editorconfig` with standard formatting rules.

- [ ] **Step 4: Relocate `Doxyfile` to `docs/Doxyfile`**
Move `Doxyfile` from repo root to `docs/Doxyfile`. Update paths inside `docs/Doxyfile` if relative paths require it (e.g. `INPUT` pointing to `include` and `src`).

- [ ] **Step 5: Untrack `docs/api-site` and update `.gitignore`**
Add `/docs/api-site/` to `.gitignore`. Remove tracked files under `docs/api-site/` via `git rm -r --cached docs/api-site`.

- [ ] **Step 6: Commit Task 1**
Run:
```bash
git add README.md CLAUDE.md AGENTS.md .clang-format .editorconfig docs/Doxyfile .gitignore
git rm docs/CLAUDE.md
git commit -m "chore: establish root directory hygiene and gitignore rules"
```

---

### Task 2: CLI Doxyfile Service Update

**Files:**
- Modify: `cli/sushiai/services/project.py:555-565`
- Test: `cli/tests/test_project.py`

**Interfaces:**
- Consumes: `docs/Doxyfile`
- Produces: `doxygen()` function in CLI resolving `docs/Doxyfile` first.

- [ ] **Step 1: Inspect and update `cli/sushiai/services/project.py`**
Modify line 558 in `cli/sushiai/services/project.py`:
```python
doxyfile = root / "docs" / "Doxyfile"
if not doxyfile.exists():
    doxyfile = root / "Doxyfile"
```

- [ ] **Step 2: Run CLI tests**
Run: `pytest cli/tests -q`
Expected: PASS (12 passed).

- [ ] **Step 3: Test `sa doxygen` execution**
Run: `sa doxygen`
Expected: Doxygen parses source tree and updates `docs/api-site` without error.

- [ ] **Step 4: Commit Task 2**
Run:
```bash
git add cli/sushiai/services/project.py
git commit -m "fix(cli): resolve Doxyfile from docs directory with root fallback"
```

---

### Task 3: Documentation Hierarchy Restructuring (`docs/`)

**Files:**
- Move: `docs/TUTORIAL.md` -> `docs/getting_started/TUTORIAL.md`
- Move: `docs/ARCHITECTURE.md` -> `docs/architecture/ARCHITECTURE.md`
- Move: `docs/CLI_GUIDE.md` -> `docs/guides/CLI_GUIDE.md`
- Move: `docs/INTEGRATION.md` -> `docs/guides/INTEGRATION.md`
- Move: `docs/CHANGELOG.md` -> `docs/reference/CHANGELOG.md`
- Move: `docs/CODE_OF_CONDUCT.md` -> `docs/reference/CODE_OF_CONDUCT.md`
- Move: `docs/SECURITY.md` -> `docs/reference/SECURITY.md`
- Move: `docs/slop/make_sushiai_great_again.md` -> `docs/design/ROADMAP.md`
- Create: `docs/design/README.md`
- Create: `docs/design/REMAINING_WORK.md`
- Create: `docs/README.md` (manual index)
- Move to archive: `docs/superpowers/plans/*` and `docs/superpowers/specs/*` -> `docs/archive/superpowers/`

**Interfaces:**
- Produces: Strict `docs/` hierarchy matching `documentation` skill.

- [ ] **Step 1: Create target directory structure under `docs/`**
Create:
- `docs/getting_started`
- `docs/architecture`
- `docs/guides`
- `docs/reference`
- `docs/design`
- `docs/archive/superpowers/plans`
- `docs/archive/superpowers/specs`

- [ ] **Step 2: Move getting started, architecture, guides, and reference documents**
Move:
- `git mv docs/TUTORIAL.md docs/getting_started/TUTORIAL.md`
- `git mv docs/ARCHITECTURE.md docs/architecture/ARCHITECTURE.md`
- `git mv docs/CLI_GUIDE.md docs/guides/CLI_GUIDE.md`
- `git mv docs/INTEGRATION.md docs/guides/INTEGRATION.md`
- `git mv docs/CODE_OF_CONDUCT.md docs/reference/CODE_OF_CONDUCT.md`
- `git mv docs/SECURITY.md docs/reference/SECURITY.md`

- [ ] **Step 3: Standardize and move `CHANGELOG.md`**
Move `docs/CHANGELOG.md` to `docs/reference/CHANGELOG.md`. Format entries to meet the 240-char single-sentence rule:
`- YYYY-MM-DD — <scope>: <Past-tense verb> <what changed> (<where>).`

- [ ] **Step 4: Move `docs/slop/make_sushiai_great_again.md` to `docs/design/ROADMAP.md`**
Move file to `docs/design/ROADMAP.md`. Ensure top line has `**Status:** Open — phase 2 of 4`. Remove empty `docs/slop/` folder.

- [ ] **Step 5: Create `docs/design/README.md` and `docs/design/REMAINING_WORK.md`**
Create `docs/design/README.md` mapping topics (`ROADMAP.md`). Create `docs/design/REMAINING_WORK.md` listing outstanding backlog items.

- [ ] **Step 6: Rewrite `docs/README.md` as Manual Index**
Write `docs/README.md` providing category links to:
- Getting Started (`getting_started/TUTORIAL.md`)
- Architecture (`architecture/ARCHITECTURE.md`)
- Guides (`guides/CLI_GUIDE.md`, `guides/INTEGRATION.md`)
- Reference (`reference/CHANGELOG.md`, `reference/CODE_OF_CONDUCT.md`, `reference/SECURITY.md`)
- Design & Roadmaps (`design/README.md`, `design/ROADMAP.md`, `design/REMAINING_WORK.md`)

- [ ] **Step 7: Archive old superpowers plans/specs**
Move past 2026-08-07 plans and specs from `docs/superpowers/` into `docs/archive/superpowers/`.

- [ ] **Step 8: Commit Task 3**
Run:
```bash
git add docs/
git commit -m "docs: restructure documentation tree into canonical layout"
```

---

### Task 4: Cross-References & Link Integrity Pass

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture/ARCHITECTURE.md`
- Modify: `docs/getting_started/TUTORIAL.md`
- Modify: `docs/guides/CLI_GUIDE.md`
- Modify: `docs/guides/INTEGRATION.md`
- Modify: `docs/CONTRIBUTING.md`
- Modify: `include/SushiAI/**/*.hpp` (where doc comments reference `docs/slop/...` or old paths)

**Interfaces:**
- Produces: 100% valid relative links across all documentation and headers.

- [ ] **Step 1: Update links in root `README.md` and `docs/CONTRIBUTING.md`**
Replace all `docs/slop/make_sushiai_great_again.md` links with `docs/design/ROADMAP.md`.
Replace all `ARCHITECTURE.md` links with `docs/architecture/ARCHITECTURE.md`.
Replace all `TUTORIAL.md` links with `docs/getting_started/TUTORIAL.md`.
Replace all `CLI_GUIDE.md` links with `docs/guides/CLI_GUIDE.md`.

- [ ] **Step 2: Update links in `docs/architecture/ARCHITECTURE.md`**
Update all relative links pointing to other docs (e.g. `../README.md`, `../guides/CLI_GUIDE.md`, `../design/ROADMAP.md`).

- [ ] **Step 3: Update links in `docs/getting_started/TUTORIAL.md` and `docs/guides/*.md`**
Adjust relative markdown links to match new folder depths.

- [ ] **Step 4: Update header docstrings referencing old paths**
Search with `grep` for `make_sushiai_great_again.md` or `slop/` across `include/SushiAI/` and `src/`, replacing with `design/ROADMAP.md`.

- [ ] **Step 5: Verify no stale link targets remain**
Run:
```powershell
Get-ChildItem -Path "docs", "include", "README.md" -Recurse -File | Select-String -Pattern "docs/slop"
```
Expected: 0 matches.

- [ ] **Step 6: Commit Task 4**
Run:
```bash
git add README.md docs/ include/
git commit -m "docs: fix cross-references and internal links for restructured paths"
```

---

### Task 5: End-to-End Verification

**Files:**
- Verification only

- [ ] **Step 1: Verify Python CLI tests**
Run: `pytest cli/tests -q`
Expected: 12 passed.

- [ ] **Step 2: Verify Doxygen documentation build**
Run: `sa doxygen`
Expected: Success, generates html into gitignored `docs/api-site/html/`.

- [ ] **Step 3: Verify C++ functional tests**
Run: `sa test --suite unit`
Expected: All unit tests pass.

- [ ] **Step 4: Verify git status hygiene**
Run: `git status --short`
Expected: Clean working tree, no untracked clutter.
