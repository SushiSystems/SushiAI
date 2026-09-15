# Task 3 Report: Documentation Hierarchy Restructuring (`docs/`)

**Date:** 2026-09-16  
**Scope:** Task 3 from `docs/superpowers/plans/2026-09-16-docs-and-root-hygiene.md`

## What was done

1. **Target directories created:**
   - Created `docs/getting_started`, `docs/architecture`, `docs/guides`, `docs/reference`, `docs/design`, `docs/archive/superpowers/plans`, and `docs/archive/superpowers/specs`.

2. **Core documents relocated:**
   - Moved `docs/TUTORIAL.md` to `docs/getting_started/TUTORIAL.md`.
   - Moved `docs/ARCHITECTURE.md` to `docs/architecture/ARCHITECTURE.md`.
   - Moved `docs/CLI_GUIDE.md` to `docs/guides/CLI_GUIDE.md`.
   - Moved `docs/INTEGRATION.md` to `docs/guides/INTEGRATION.md`.
   - Moved `docs/CODE_OF_CONDUCT.md` to `docs/reference/CODE_OF_CONDUCT.md`.
   - Moved `docs/SECURITY.md` to `docs/reference/SECURITY.md`.

3. **`CHANGELOG.md` standardized and relocated:**
   - Moved `docs/CHANGELOG.md` to `docs/reference/CHANGELOG.md`.
   - Replaced multi-paragraph milestone descriptions with standardized single-sentence entries under `## Unreleased` conforming to the `documentation` skill:
     `- <YYYY-MM-DD> — <scope>: <Past-tense verb> <what changed> (<where>).`
   - Verified that every entry is at most 240 characters and names at most five backticked locations.

4. **Roadmap relocated and status line updated:**
   - Moved `docs/slop/make_sushiai_great_again.md` to `docs/design/ROADMAP.md`.
   - Set line 3 status line to `**Status:** Open — phase 2 of 4`.
   - Deleted the empty directory `docs/slop`.

5. **Historical superpowers plans and specs archived:**
   - Moved `docs/superpowers/plans/2026-08-07*.md` to `docs/archive/superpowers/plans/`.
   - Moved `docs/superpowers/specs/2026-08-07*.md` to `docs/archive/superpowers/specs/`.
   - Retained active `2026-09-16` files in `docs/superpowers/`.

6. **`docs/design/README.md` created:**
   - Created the design document index table mapping `ROADMAP.md` to the SushiBLAS & SushiAI Construction & Roadmap topic.

7. **`docs/design/REMAINING_WORK.md` created:**
   - Created the repository backlog detailing open technical debt: scratch memory arena packing, batch copy overhead, row-wise log-softmax fusion, standalone log-softmax backward IR node, whole-step loss scaling skip, dataset interface generalization beyond classification, arbitrary objective leaves, and demo metric regression tracking.

8. **`docs/README.md` rewritten as manual index:**
   - Replaced duplicate documentation content with a manual index linking Getting Started, Architecture, Guides, Reference, Design & Roadmaps, and Contributing.

## What was not done

- Did not commit changes to git (deferred to git orchestrator).
- Did not modify files outside `docs/`.
- Did not update internal relative link references in documentation bodies or C++ headers (assigned to Task 4).
- Did not run the build system or modify CMake build scripts.

## Verifications and outputs

### 1. Git status of documentation reorganization

Command:
```powershell
git status --short
```

Output:
```text
M  docs/README.md
R  docs/ARCHITECTURE.md -> docs/architecture/ARCHITECTURE.md
R  docs/superpowers/plans/2026-08-07-config-driven-training-plan.md -> docs/archive/superpowers/plans/2026-08-07-config-driven-training-plan.md
R  docs/superpowers/plans/2026-08-07-cv-blas-spatial-plan.md -> docs/archive/superpowers/plans/2026-08-07-cv-blas-spatial-plan.md
R  docs/superpowers/plans/2026-08-07-cv-ir-autograd-plan.md -> docs/archive/superpowers/plans/2026-08-07-cv-ir-autograd-plan.md
R  docs/superpowers/specs/2026-08-07-config-driven-training-design.md -> docs/archive/superpowers/specs/2026-08-07-config-driven-training-design.md
R  docs/superpowers/specs/2026-08-07-cv-pipeline-layers-design.md -> docs/archive/superpowers/specs/2026-08-07-cv-pipeline-layers-design.md
A  docs/design/README.md
A  docs/design/REMAINING_WORK.md
RM docs/slop/make_sushiai_great_again.md -> docs/design/ROADMAP.md
R  docs/TUTORIAL.md -> docs/getting_started/TUTORIAL.md
R  docs/CLI_GUIDE.md -> docs/guides/CLI_GUIDE.md
R  docs/INTEGRATION.md -> docs/guides/INTEGRATION.md
RM docs/CHANGELOG.md -> docs/reference/CHANGELOG.md
R  docs/CODE_OF_CONDUCT.md -> docs/reference/CODE_OF_CONDUCT.md
R  docs/SECURITY.md -> docs/reference/SECURITY.md
```

### 2. Line length check on `docs/reference/CHANGELOG.md`

Command:
```powershell
Get-Content "docs\reference\CHANGELOG.md" | Where-Object { $_.Length -gt 240 }
```

Output:
```text
(0 lines returned)
```

### 3. Verification of all relative links from `docs/README.md`

Command:
```powershell
$links = @(
    "docs\getting_started\TUTORIAL.md",
    "docs\architecture\ARCHITECTURE.md",
    "docs\guides\CLI_GUIDE.md",
    "docs\guides\INTEGRATION.md",
    "docs\reference\CHANGELOG.md",
    "docs\reference\CODE_OF_CONDUCT.md",
    "docs\reference\SECURITY.md",
    "docs\design\README.md",
    "docs\design\ROADMAP.md",
    "docs\design\REMAINING_WORK.md",
    "docs\CONTRIBUTING.md"
)
foreach ($l in $links) {
    [PSCustomObject]@{ Path = $l; Exists = (Test-Path $l) }
}
```

Output:
```text
Path                              Exists
----                              ------
docs\getting_started\TUTORIAL.md    True
docs\architecture\ARCHITECTURE.md   True
docs\guides\CLI_GUIDE.md            True
docs\guides\INTEGRATION.md          True
docs\reference\CHANGELOG.md         True
docs\reference\CODE_OF_CONDUCT.md   True
docs\reference\SECURITY.md          True
docs\design\README.md               True
docs\design\ROADMAP.md              True
docs\design\REMAINING_WORK.md       True
docs\CONTRIBUTING.md                True
```

### 4. Removal of `docs/slop` directory

Command:
```powershell
Test-Path "docs\slop"
```

Output:
```text
False
```

### 5. Archived superpower files

Command:
```powershell
Get-ChildItem -Path "docs\archive\superpowers" -Recurse -File | Select-Object FullName
```

Output:
```text
FullName                                                                                      
--------                                                                                      
D:\Projects\sushiai\docs\archive\superpowers\plans\2026-08-07-config-driven-training-plan.md  
D:\Projects\sushiai\docs\archive\superpowers\plans\2026-08-07-cv-blas-spatial-plan.md         
D:\Projects\sushiai\docs\archive\superpowers\plans\2026-08-07-cv-ir-autograd-plan.md          
D:\Projects\sushiai\docs\archive\superpowers\specs\2026-08-07-config-driven-training-design.md
D:\Projects\sushiai\docs\archive\superpowers\specs\2026-08-07-cv-pipeline-layers-design.md    
```
