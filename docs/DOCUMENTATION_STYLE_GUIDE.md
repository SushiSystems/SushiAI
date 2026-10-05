# Documentation Style Guide

This guide establishes the rules for writing, structuring, and maintaining documentation in SushiAI. Every document in the repository must adhere to this guide. Automated checkers under `tools/documentation/` enforce these rules before commits land.

## 1. Directory Structure

The `docs/` tree holds all project documentation outside code-level READMEs. Each kind of fact has exactly one location:

```
docs/
    README.md                     manual index; every live document is linked here
    CONTRIBUTING.md               contribution process, coding bar, pull request rules
    DOCUMENTATION_STYLE_GUIDE.md  this guide
    getting_started/              step-by-step onboarding and tutorials
    architecture/                 system design, IR passes, execution model
    guides/                       user and operator guides (CLI, integration)
    reference/                    changelog, glossary, known issues, security, code of conduct
    design/                       active design documents and the single backlog
    agent/                        dated work folders for automated agent workflows
    archive/                      frozen artifacts and historical plans
```

No other directories exist under `docs/`. The root holds only the files the `repository-layout` skill names.

## 2. File Naming Conventions

- Live documents use `UPPER_SNAKE_CASE.md` (e.g. `ARCHITECTURE.md`, `CLI_GUIDE.md`, `KNOWN_ISSUES.md`).
- Topic map files are named `README.md` within their respective section (e.g. `docs/design/README.md`).
- Agent work folders use `YYYY_MM_DD_WORK_NAME` format (e.g. `docs/agent/2026_09_16_DOCS_AND_ROOT_HYGIENE/`).
- Inside an agent work folder, only three files are permitted: `SPEC.md`, `PLAN.md`, and `REPORT.md`.

## 3. Placement Rules

To determine where new content belongs, evaluate these questions in order:

1. **Is it a fact specific to one module?** Place it in that module's `README.md`.
2. **Was it generated during an active agent work session?** Place it in `docs/agent/<YYYY_MM_DD>_<WORK_NAME>/`.
3. **Is it an active proposal or phased technical design?** Place it in `docs/design/<TOPIC>.md`.
4. **Is it a completed record idle for over 90 days?** Move it to `docs/archive/`.
5. **Is it permanent reference material?** Place it under the manual: `getting_started/`, `architecture/`, `guides/`, or `reference/`.

## 4. Design Documents and Backlog

- `docs/design/README.md` maintains a one-to-one mapping between technical topics and design documents. Check this index before authoring a new document. Never create duplicate documents for the same topic.
- Every design document carries its status on the first line under its title:
  ```markdown
  **Status:** Open — phase 2 of 4
  ```
  or:
  ```markdown
  **Status:** Shipped
  ```
- Design documents exceeding 1,500 lines must be split into a topic directory containing a `README.md` index and one file per phase.
- `docs/design/REMAINING_WORK.md` is the repository's sole backlog. Do not maintain separate TODO files or untracked task lists.

## 5. Changelog Entries

`docs/reference/CHANGELOG.md` records user-visible changes grouped by release. Entries follow strict formatting constraints:

- Format: `- <YYYY-MM-DD> — <scope>: <Past-tense verb> <what changed> (<where>).`
- Maximum length: 240 characters per line.
- Single sentence, ending with a period.
- At most five backticked file paths or symbol names in parentheses.
- Newest entry first within each section.
- Breaking changes must explicitly state what was altered or removed.

## 6. Prose Standards

- Use clear, declarative statements in active voice.
- State exact facts, measurements, and identifiers rather than unquantified intensifiers ("significantly faster", "robust").
- Conclude immediately once the factual content is delivered. Do not append decorative summary paragraphs or filler sign-offs.
- Turkish prose must place the verb at the end of the sentence, avoid inverted sentence structure, and avoid em dashes except as dialogue indicators.

## 7. Link Integrity

- Every relative Markdown link must resolve to a valid file on disk.
- Never use broken anchors or obsolete file paths.
- Code blocks containing path examples must use code spans or fenced code blocks, not live Markdown link syntax.

## 8. Verification

Run the documentation checkers before committing changes to `docs/`:

```bash
python tools/documentation/check_docs_layout.py .
python tools/documentation/check_changelog.py .
```
