# Backlog migration: inventory

**Status:** Shipped — 2026-10-19. The owner approved the inventory as proposed.

The migration of this repository to the backlog format of SushiSkills'
`skills/documentation/BACKLOG_FORMAT.md`, step 2 of section 8 of SushiSkills'
`docs/agent/2026_10_19_BACKLOG_FORMAT/SPEC.md`.

## Fact check

Both files were last written at `5395456` (2026-10-17); the commits since change no code the
entries name. Each entry was read against that state, and the paths written before the module
tree of 2026-10-09 were resolved to where the files are now. What changed:

| Entry | Verdict |
| --- | --- |
| "What a build has confirmed" | Finished work; its three open runs become `RW-8` to `RW-10` |
| Build and run the Darknet training parity tests, "never compiled" | Closed: YOLOv3 ran on 2026-10-12, YOLOv4 and YOLOv7 on 2026-10-13 (`bebbac6`); what they found is `KI-6` to `KI-8` |
| Make a contiguous `K_RESHAPE` a view | Closed: recorded on 2026-10-13 (`21a0470`) |
| Carry the domain-tier key of the layering checker to SushiSkills | Closed: SushiSkills' checker reads `K_DOMAIN_TIERS` |
| Let CI reach the siblings | Closed: the jobs check SushiRuntime and SushiLog out with a token |
| Make the root README a front door | Closed: cut to six sections in `7aebd6b` |
| Module READMEs | Closed: every module folder has one |
| Humanizer pass over the manual | Closed: the manual passes the em dash rule; the roadmap leaves the manual |
| Tie the CLI version to the library version | Closed: both declare 0.1.0 |
| Add `.gitattributes` | Closed: the file exists |
| `sa test --jobs 1` runs ctest with its own default | Closed: sushicore 0.9.1, pinned since `019f742`, passes a count of 1 |
| A comment cites `docs/architecture/ARCHITECTURE.md` as if it were elsewhere | Closed: the path it cites exists |
| The demo transcript in `README.md` | Moved with the transcript to `docs/guides/DEMO.md`; `KI-10` |
| The security page's links | Stale: the sibling policies are under `.github/` now; `KI-11` |
| Decide who owns the roadmap | Closed by moving `ROADMAP.md` to the archive (below) |
| The FFT backend question of the roadmap | Foreign: `sushiblas RW-6` |
| "Estate programmes" | Not work; deleted |

## Shape

- The roadmap's name is one the documentation skill no longer allows under `docs/design/`, and
  its status said it was superseded by `PROGRAMME.md` on 2026-10-07. It moved to
  `docs/archive/design/ROADMAP.md` with the status `Superseded — 2026-10-07`. The two rows that
  listed it in `docs/README.md` and the design map went; the architecture page, the getting
  started page and the root README now point at `PROGRAMME.md` for what is planned; the
  programme page and the examples page point at the archived file.
- Six open design documents are each cited by at least one entry.
- `RW-23` and `RW-28` are unused: `Scope`'s destructor joined `RW-11`, and the reshape view is
  closed.

The backlog holds `RW-1` to `RW-129` less those two, seven of them decisions; the known issues
hold `KI-1` to `KI-16`.

## Entries from and for sibling repositories

| Entry | Repository | State |
| --- | --- | --- |
| `RW-121`, move onto SushiBLAS's views | From sushiblas `RW-91` | Added here; `RW-91` there goes when SushiBLAS's current work is done |
| `RW-122`, a `Status` code for a busy runtime | From the runtime's backlog before its migration | Added here |
| R1 of the graph capabilities plan | sushiruntime | Added there as `RW-62`; R2 is the deferred decision `RW-4` here |
| The B items of the graph capabilities plan, and a bfloat16 `DataType` | sushiblas | Pending: an agent is working in that repository; the entries are added when it is done, and `RW-94`, `RW-109` and `RW-120` then cite them by ID |

## Status lines

Seventeen work folder files took the four forms. Open: `CV_CLOSE`, `EAGER_FRONT_END`,
`ERROR_CONTRACT_AND_NAMING`, `YOLO26`, `DARKNET_YOLO`, `IMPORT_DATASETS_LANGUAGE`. Superseded:
`LOGGING_FRONT`, by `CORE_SPLIT`. The rest shipped.

## Outcome

The owner approved the inventory on 2026-10-19.

```
$ python tools/check_all.py .
clean: 11 checkers, tests
```
