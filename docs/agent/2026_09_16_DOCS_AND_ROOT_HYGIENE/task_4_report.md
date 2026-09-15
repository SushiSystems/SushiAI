# Task 4 Report: Cross-References & Link Integrity Pass

**Date:** 2026-09-16  
**Scope:** Task 4 from `docs/superpowers/plans/2026-09-16-docs-and-root-hygiene.md`

## What was done

1. **Updated documentation cross-references in `docs/architecture/ARCHITECTURE.md`:**
   - Updated link to roadmap from `docs/slop/make_sushiai_great_again.md` (`slop/make_sushiai_great_again.md`) to `ROADMAP.md` (`../design/ROADMAP.md`).
   - Updated relative links to root `README.md` from `../README.md` and `README.md` to `../../README.md` (lines 11, 689, 948).
   - Updated relative link to `INTEGRATION.md` from `INTEGRATION.md#3-what-you-inherit-and-what-you-must-supply` to `../guides/INTEGRATION.md#3-what-you-inherit-and-what-you-must-supply` (line 87).
   - Replaced mention of `docs/slop` with `docs/design` (line 516).

2. **Updated documentation cross-references in `docs/getting_started/TUTORIAL.md`:**
   - Updated relative links to `CONTRIBUTING.md` from `CONTRIBUTING.md` to `../CONTRIBUTING.md` (lines 12, 493).
   - Updated relative link to root `README.md` from `README.md` to `../../README.md` (line 490).
   - Updated relative link to `ARCHITECTURE.md` from `ARCHITECTURE.md` to `../architecture/ARCHITECTURE.md` (line 491).
   - Updated relative link to `CLI_GUIDE.md` from `CLI_GUIDE.md` to `../guides/CLI_GUIDE.md` (line 492).
   - Updated relative link to `CHANGELOG.md` from `CHANGELOG.md` to `../reference/CHANGELOG.md` (line 494).

3. **Updated documentation cross-references in `docs/guides/CLI_GUIDE.md`:**
   - Updated relative link to `CONTRIBUTING.md` from `CONTRIBUTING.md#3-building-and-testing-your-change` to `../CONTRIBUTING.md#3-building-and-testing-your-change` (line 124).
   - Updated relative link to `ARCHITECTURE.md` from `ARCHITECTURE.md` to `../architecture/ARCHITECTURE.md#62-the-pass-order-is-load-bearing` (line 263).
   - Updated relative links to root `README.md` from `../README.md` to `../../README.md` (lines 277, 302, 452).

4. **Updated documentation cross-references in `docs/guides/INTEGRATION.md`:**
   - Updated relative link to `ARCHITECTURE.md §3` from `ARCHITECTURE.md#3-the-library-target` to `../architecture/ARCHITECTURE.md#3-the-library-target` (line 34).
   - Updated relative link to `D3 layered resolution` from `ARCHITECTURE.md#2-the-sibling-dependency-chain-d3` to `../architecture/ARCHITECTURE.md#2-the-sibling-dependency-chain-d3` (line 68).
   - Updated relative link to `ARCHITECTURE.md §1` from `ARCHITECTURE.md#the-whole-stack-is-c17-and-that-is-not-a-style-choice` to `../architecture/ARCHITECTURE.md#the-whole-stack-is-c17-and-that-is-not-a-style-choice` (line 108).

5. **Updated documentation cross-references in `docs/CONTRIBUTING.md`:**
   - Updated relative link to `TUTORIAL.md` from `TUTORIAL.md` to `getting_started/TUTORIAL.md` (line 15).
   - Updated relative links to `ARCHITECTURE.md` from `ARCHITECTURE.md` to `architecture/ARCHITECTURE.md` (lines 17, 138, 167, 180).
   - Updated relative link to roadmap from `docs/slop/make_sushiai_great_again.md` (`slop/make_sushiai_great_again.md`) to `docs/design/ROADMAP.md` (`design/ROADMAP.md`) (line 18).
   - Updated relative links to `CLI_GUIDE.md` from `CLI_GUIDE.md` to `guides/CLI_GUIDE.md` (lines 43, 182).
   - Updated relative link to `CHANGELOG.md` from `CHANGELOG.md` to `reference/CHANGELOG.md` (line 177).
   - Updated relative links to `SECURITY.md` from `SECURITY.md` to `reference/SECURITY.md` (lines 242, 249).
   - Updated relative link to `CODE_OF_CONDUCT.md` from `CODE_OF_CONDUCT.md` to `reference/CODE_OF_CONDUCT.md` (line 253).

6. **Verified `README.md` and `docs/design/ROADMAP.md`:**
   - Verified that all 22 markdown links in root `README.md` resolve to existing files (`docs/getting_started/TUTORIAL.md`, `docs/architecture/ARCHITECTURE.md`, `docs/guides/CLI_GUIDE.md`, `docs/guides/INTEGRATION.md`, `docs/design/ROADMAP.md`, `LICENSE`).
   - Verified that `docs/design/ROADMAP.md` contains no broken markdown links.

7. **Updated C++ headers under `include/SushiAI/`:**
   - `include/SushiAI/optim/loss_scaler.hpp`: replaced `@c docs/slop/make_sushiai_great_again.md §21` with `@c docs/design/ROADMAP.md §21` (line 89).
   - `include/SushiAI/config/registry.hpp`: replaced `docs/ARCHITECTURE.md §4` with `docs/architecture/ARCHITECTURE.md §4` (line 44).
   - `include/SushiAI/train/trainer.hpp`: replaced `docs/ARCHITECTURE.md` with `docs/architecture/ARCHITECTURE.md` (line 164).
   - `include/SushiAI/train/objective.hpp`: replaced `docs/ARCHITECTURE.md` with `docs/architecture/ARCHITECTURE.md` (line 57).
   - `include/SushiAI/nn/module.hpp`: replaced `docs/ARCHITECTURE.md §1` with `docs/architecture/ARCHITECTURE.md §1` (line 402).
   - `include/SushiAI/nn/ilayer.hpp`: replaced `docs/ARCHITECTURE.md §1` with `docs/architecture/ARCHITECTURE.md §1` (line 41).

## What was not done

- Did not commit changes to git.
- Did not run the build system or edit build configuration files (`CMakeLists.txt`) per Rule 3.
- Did not modify files outside the allowed target list (`examples/heavy_mlp/README.md`, `src/CMakeLists.txt`, and archived superpower plans) per Rule 4.

## Verifications and outputs

### 1. Git status of modified files

Command:
```powershell
git status --short
```

Output:
```
 M docs/CONTRIBUTING.md
 M docs/architecture/ARCHITECTURE.md
 M docs/getting_started/TUTORIAL.md
 M docs/guides/CLI_GUIDE.md
 M docs/guides/INTEGRATION.md
 M include/SushiAI/config/registry.hpp
 M include/SushiAI/nn/ilayer.hpp
 M include/SushiAI/nn/module.hpp
 M include/SushiAI/optim/loss_scaler.hpp
 M include/SushiAI/train/objective.hpp
 M include/SushiAI/train/trainer.hpp
```

### 2. Comprehensive documentation link verification script

Command:
```python
import os, re

files_to_check = [
    'README.md',
    'docs/CONTRIBUTING.md',
    'docs/README.md',
    'docs/architecture/ARCHITECTURE.md',
    'docs/getting_started/TUTORIAL.md',
    'docs/guides/CLI_GUIDE.md',
    'docs/guides/INTEGRATION.md',
    'docs/design/ROADMAP.md',
    'docs/reference/CHANGELOG.md',
    'docs/reference/CODE_OF_CONDUCT.md',
    'docs/reference/SECURITY.md'
]

link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')

total_links = 0
passed_links = 0
failed_links = 0

for file_rel in files_to_check:
    if not os.path.exists(file_rel):
        print(f'FILE NOT FOUND: {file_rel}')
        continue
    content = open(file_rel, encoding='utf-8').read()
    matches = link_pattern.findall(content)
    print(f'File: {file_rel} ({len(matches)} links)')
    for text, url in matches:
        if url.startswith(('http://', 'https://', 'mailto:')):
            print(f'  [EXTERNAL] [{text}] -> {url}')
            continue
        target_file = url.split('#')[0]
        anchor = url.split('#')[1] if '#' in url else None
        if not target_file:
            print(f'  [ANCHOR] [{text}] -> #{anchor}')
            continue
        base_dir = os.path.dirname(file_rel)
        resolved = os.path.normpath(os.path.join(base_dir, target_file))
        exists = os.path.exists(resolved)
        total_links += 1
        if exists:
            passed_links += 1
            print(f'  [PASS] [{text}]({url}) -> {resolved}')
        else:
            failed_links += 1
            print(f'  [FAIL] [{text}]({url}) -> {resolved} (NOT FOUND)')

print(f'\nSummary: Total local file links checked: {total_links}, Passed: {passed_links}, Failed: {failed_links}')
```

Output:
```
File: README.md (22 links)
  [EXTERNAL] [SushiBLAS] -> https://github.com/SushiSystems/SushiBLAS
  [EXTERNAL] [SushiRuntime] -> https://github.com/SushiSystems/SushiRuntime
  [PASS] [`docs/design/ROADMAP.md`](docs/design/ROADMAP.md) -> docs\design\ROADMAP.md
  [PASS] [ARCHITECTURE.md §7](docs/architecture/ARCHITECTURE.md#7-whats-not-here-yet) -> docs\architecture\ARCHITECTURE.md
  [PASS] [`TUTORIAL.md`](docs/getting_started/TUTORIAL.md) -> docs\getting_started\TUTORIAL.md
  [PASS] [ARCHITECTURE.md §5.1](docs/architecture/ARCHITECTURE.md#51-the-optimizer-is-part-of-the-compiled-step) -> docs\architecture\ARCHITECTURE.md
  [PASS] [CLI_GUIDE.md](docs/guides/CLI_GUIDE.md#sushiai-train-and-sushiai-inference--config-driven-runs) -> docs\guides\CLI_GUIDE.md
  [ANCHOR] [the demo section] -> #the-demo-sa-demo-mlp
  [PASS] [D3-resolved](docs/architecture/ARCHITECTURE.md#2-the-sibling-dependency-chain-d3) -> docs\architecture\ARCHITECTURE.md
  [EXTERNAL] [SushiEngine] -> https://github.com/SushiSystems/SushiEngine
  [PASS] [D3 layered CMake resolution](docs/architecture/ARCHITECTURE.md#2-the-sibling-dependency-chain-d3) -> docs\architecture\ARCHITECTURE.md
  [PASS] [ARCHITECTURE.md §1](docs/architecture/ARCHITECTURE.md#the-whole-stack-is-c17-and-that-is-not-a-style-choice) -> docs\architecture\ARCHITECTURE.md
  [PASS] [INTEGRATION.md §3](docs/guides/INTEGRATION.md#3-what-you-inherit-and-what-you-must-supply) -> docs\guides\INTEGRATION.md
  [EXTERNAL] [SushiStack] -> https://sushisystems.io
  [EXTERNAL] [SushiStack] -> https://sushisystems.io
  [PASS] [INTEGRATION.md](docs/guides/INTEGRATION.md) -> docs\guides\INTEGRATION.md
  [PASS] [D3 resolution](docs/architecture/ARCHITECTURE.md#2-the-sibling-dependency-chain-d3) -> docs\architecture\ARCHITECTURE.md
  [PASS] [ARCHITECTURE.md §5.1](docs/architecture/ARCHITECTURE.md#51-the-optimizer-is-part-of-the-compiled-step) -> docs\architecture\ARCHITECTURE.md
  [PASS] [§5.2](docs/architecture/ARCHITECTURE.md#52-fusion-is-a-selector-over-a-catalog-not-a-code-generator) -> docs\architecture\ARCHITECTURE.md
  [PASS] [ARCHITECTURE.md](docs/architecture/ARCHITECTURE.md) -> docs\architecture\ARCHITECTURE.md
  [PASS] [the master plan](docs/design/ROADMAP.md) -> docs\design\ROADMAP.md
  [PASS] [LICENSE](LICENSE) -> LICENSE
File: docs/CONTRIBUTING.md (18 links)
  [PASS] [`LICENSE`](../LICENSE) -> LICENSE
  [PASS] [`TUTORIAL.md`](getting_started/TUTORIAL.md) -> docs\getting_started\TUTORIAL.md
  [PASS] [`ARCHITECTURE.md`](architecture/ARCHITECTURE.md) -> docs\architecture\ARCHITECTURE.md
  [PASS] [`docs/design/ROADMAP.md`](design/ROADMAP.md) -> docs\design\ROADMAP.md
  [PASS] [`CLI_GUIDE.md`](guides/CLI_GUIDE.md) -> docs\guides\CLI_GUIDE.md
  [PASS] [ARCHITECTURE.md §1](architecture/ARCHITECTURE.md#the-whole-stack-is-c17-and-that-is-not-a-style-choice) -> docs\architecture\ARCHITECTURE.md
  [EXTERNAL] [`CONTRIBUTING.md` §4] -> https://github.com/SushiSystems/SushiBLAS/blob/main/docs/CONTRIBUTING.md#4-coding-style
  [PASS] [ARCHITECTURE.md §4](architecture/ARCHITECTURE.md#4-packaging) -> docs\architecture\ARCHITECTURE.md
  [PASS] [`CHANGELOG.md`](reference/CHANGELOG.md) -> docs\reference\CHANGELOG.md
  [PASS] [`ARCHITECTURE.md`](architecture/ARCHITECTURE.md) -> docs\architecture\ARCHITECTURE.md
  [PASS] [`CLI_GUIDE.md`](guides/CLI_GUIDE.md) -> docs\guides\CLI_GUIDE.md
  [PASS] [`cli/sushistack.deps.toml`](../cli/sushistack.deps.toml) -> cli\sushistack.deps.toml
  [ANCHOR] [§7] -> #7-adding-a-c-dependency
  [EXTERNAL] [SushiStack] -> https://sushisystems.io
  [PASS] [`cli/sushistack.deps.toml`](../cli/sushistack.deps.toml) -> cli\sushistack.deps.toml
  [PASS] [`SECURITY.md`](reference/SECURITY.md) -> docs\reference\SECURITY.md
  [PASS] [`SECURITY.md`](reference/SECURITY.md) -> docs\reference\SECURITY.md
  [PASS] [`CODE_OF_CONDUCT.md`](reference/CODE_OF_CONDUCT.md) -> docs\reference\CODE_OF_CONDUCT.md
File: docs/README.md (11 links)
  [PASS] [Tutorial](getting_started/TUTORIAL.md) -> docs\getting_started\TUTORIAL.md
  [PASS] [Architecture Overview](architecture/ARCHITECTURE.md) -> docs\architecture\ARCHITECTURE.md
  [PASS] [CLI Guide](guides/CLI_GUIDE.md) -> docs\guides\CLI_GUIDE.md
  [PASS] [Integration Guide](guides/INTEGRATION.md) -> docs\guides\INTEGRATION.md
  [PASS] [Changelog](reference/CHANGELOG.md) -> docs\reference\CHANGELOG.md
  [PASS] [Code of Conduct](reference/CODE_OF_CONDUCT.md) -> docs\reference\CODE_OF_CONDUCT.md
  [PASS] [Security Policy](reference/SECURITY.md) -> docs\reference\SECURITY.md
  [PASS] [Design Map](design/README.md) -> docs\design\README.md
  [PASS] [SushiStack Construction & Roadmap](design/ROADMAP.md) -> docs\design\ROADMAP.md
  [PASS] [Remaining Work](design/REMAINING_WORK.md) -> docs\design\REMAINING_WORK.md
  [PASS] [Contributing Guide](CONTRIBUTING.md) -> docs\CONTRIBUTING.md
File: docs/architecture/ARCHITECTURE.md (8 links)
  [PASS] [`ROADMAP.md`](../design/ROADMAP.md) -> docs\design\ROADMAP.md
  [PASS] [README.md](../../README.md) -> README.md
  [EXTERNAL] [SushiBLAS] -> https://github.com/SushiSystems/SushiBLAS
  [EXTERNAL] [SushiRuntime] -> https://github.com/SushiSystems/SushiRuntime
  [EXTERNAL] [its ARCHITECTURE §1] -> https://github.com/SushiSystems/SushiBLAS/blob/main/docs/ARCHITECTURE.md#1-what-sushiblas-is
  [PASS] [INTEGRATION.md §3](../guides/INTEGRATION.md#3-what-you-inherit-and-what-you-must-supply) -> docs\guides\INTEGRATION.md
  [PASS] [README.md § Mixed precision](../../README.md#mixed-precision-what---precision-mixed-does-and-what-it-is-worth-here) -> README.md
  [PASS] [README.md § Project layout](../../README.md#project-layout) -> README.md
File: docs/getting_started/TUTORIAL.md (6 links)
  [PASS] [CONTRIBUTING.md](../CONTRIBUTING.md) -> docs\CONTRIBUTING.md
  [PASS] [README.md](../../README.md) -> README.md
  [PASS] [ARCHITECTURE.md](../architecture/ARCHITECTURE.md) -> docs\architecture\ARCHITECTURE.md
  [PASS] [CLI_GUIDE.md](../guides/CLI_GUIDE.md) -> docs\guides\CLI_GUIDE.md
  [PASS] [CONTRIBUTING.md](../CONTRIBUTING.md) -> docs\CONTRIBUTING.md
  [PASS] [CHANGELOG.md](../reference/CHANGELOG.md) -> docs\reference\CHANGELOG.md
File: docs/guides/CLI_GUIDE.md (7 links)
  [EXTERNAL] [SushiEngine] -> https://github.com/SushiSystems/SushiEngine
  [PASS] [CONTRIBUTING.md §3](../CONTRIBUTING.md#3-building-and-testing-your-change) -> docs\CONTRIBUTING.md
  [PASS] [INTEGRATION.md](INTEGRATION.md) -> docs\guides\INTEGRATION.md
  [PASS] [ARCHITECTURE.md §6.2](../architecture/ARCHITECTURE.md#62-the-pass-order-is-load-bearing) -> docs\architecture\ARCHITECTURE.md
  [PASS] [README.md § Mixed precision](../../README.md#mixed-precision-what---precision-mixed-does-and-what-it-is-worth-here) -> README.md
  [PASS] [README.md § The demo](../../README.md#the-demo-sa-demo-mlp) -> README.md
  [PASS] [README.md](../../README.md#building-with-cmake-directly) -> README.md
File: docs/guides/INTEGRATION.md (4 links)
  [EXTERNAL] [`INTEGRATION.md` §1] -> https://github.com/SushiSystems/SushiBLAS/blob/main/docs/INTEGRATION.md#1-the-fact-that-shapes-everything
  [PASS] [ARCHITECTURE.md §3](../architecture/ARCHITECTURE.md#3-the-library-target) -> docs\architecture\ARCHITECTURE.md
  [PASS] [D3 layered resolution](../architecture/ARCHITECTURE.md#2-the-sibling-dependency-chain-d3) -> docs\architecture\ARCHITECTURE.md
  [PASS] [ARCHITECTURE.md §1](../architecture/ARCHITECTURE.md#the-whole-stack-is-c17-and-that-is-not-a-style-choice) -> docs\architecture\ARCHITECTURE.md
File: docs/design/ROADMAP.md (0 links)
File: docs/reference/CHANGELOG.md (0 links)
File: docs/reference/CODE_OF_CONDUCT.md (0 links)
File: docs/reference/SECURITY.md (2 links)
  [EXTERNAL] [SushiBLAS] -> https://github.com/SushiSystems/SushiBLAS/blob/main/docs/SECURITY.md
  [EXTERNAL] [SushiRuntime] -> https://github.com/SushiSystems/SushiRuntime/blob/main/docs/SECURITY.md

Summary: Total local file links checked: 62, Passed: 62, Failed: 0
```
