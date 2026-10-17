# log

## Owns

SushiAI's front on SushiLog, the one logger of the process. `log.hpp` defines the `SA_LOG_*`
macros on `SL_LOG_FRONT_AT`, with the ceiling `K_SUSHIAI_LOG_CEILING` that each translation unit
computes from `SA_ENABLE_LOGGING`. `K_SUSHIAI_LOG_PROJECT` names the project `sushiai`, whose
level `SA_LOG_LEVEL` sets without touching another project's. The module writes no logger, sink,
queue or thread of its own. Code that sets a sink or a level calls `SushiLog::` directly.

A module that logs declares its own category in `source/<module>_log_category.hpp`, made with
`SushiLog::make_log_category(K_SUSHIAI_LOG_PROJECT, "<module>")`, and logs only under it.

## Depends on

The module declares no module dependency. It links `SushiLog::SushiLog` as public, through
`EXTERNAL_PUBLIC`.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/log/log.hpp` | `SA_ENABLE_LOGGING`, `K_SUSHIAI_LOG_CEILING`, `SA_LOG_AT`, `SA_LOG_TRACE`, `SA_LOG_DEBUG`, `SA_LOG_INFO`, `SA_LOG_WARN`, `SA_LOG_ERROR`, `SA_LOG_FATAL` |
| `SushiAI/log/sushiai_log_project.hpp` | `K_SUSHIAI_LOG_PROJECT` |
