# optim

## Owns

The optimizers and the state they keep across a training step. `Optimizer` is the abstract base.
It holds the `ParameterSlot`s of a model and is a `Graph::StepRecorder`, so it records its
update into a step. `SGD`, with momentum, and `AdamW`, with decoupled weight decay, are the two
implementations. `Optimizer::learning_rate` and `set_learning_rate` read and set the step size of
either one through the base class. `StepScalars` holds the scalars that the update kernels read from device
memory. `LossScaler` applies dynamic loss scaling for mixed precision, and `GradientClip`
clips a step's gradients to a global norm. `StepSchedule` and `CosineSchedule` give the rate
of each update. `optimizer_registry` maps
a configured type name to a factory through `K_OPTIMIZER_REGISTRY`.

## Depends on

The module declares `core`, `graph` and `config` as public dependencies. It declares no private
dependency.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/optim/adamw.hpp` | `AdamWOptions`, `AdamW` |
| `SushiAI/optim/apply_step_schedule.hpp` | `apply_step_schedule`, over one schedule type and over a `RateSchedule` |
| `SushiAI/optim/cosine_schedule.hpp` | `CosineSchedule`, `validate_cosine_schedule` |
| `SushiAI/optim/cosine_schedule_section.hpp` | `K_COSINE_SCHEDULE_TYPE`, `read_cosine_schedule_section`, `write_cosine_schedule_section` |
| `SushiAI/optim/gradient_clip.hpp` | `GradientClip` |
| `SushiAI/optim/loss_scaler.hpp` | `LossScaleOptions`, `LossScaleState`, `LossScaler` |
| `SushiAI/optim/optimizer.hpp` | `UpdateKernel`, `ParameterSlot`, `validate_slots`, `Optimizer` |
| `SushiAI/optim/optimizer_registry.hpp` | `OptimizerFactory`, `OptimizerEntry`, `K_OPTIMIZER_REGISTRY`, `make_adamw`, `make_sgd`, `make_optimizer` |
| `SushiAI/optim/rate_schedule.hpp` | `RateSchedule`, `rate_at`, `base_rate_of`, `validate_rate_schedule` |
| `SushiAI/optim/rate_schedule_section.hpp` | `read_rate_schedule_section`, `write_rate_schedule_section` |
| `SushiAI/optim/run_schedule.hpp` | `K_LEARNING_RATE_KEY`, `read_rate_schedule` |
| `SushiAI/optim/run_state.hpp` | `K_STEP_COUNT_STATE`, `K_LOSS_SCALE_STATE`, `K_LOSS_SCALE_GOOD_STEPS_STATE`, `K_MOMENTUM_ROLE`, `K_FIRST_MOMENT_ROLE`, `K_SECOND_MOMENT_ROLE`, `StateTensor`, `StateScalar`, `RunState` |
| `SushiAI/optim/sgd.hpp` | `SGDOptions`, `SGD` |
| `SushiAI/optim/step_scalars.hpp` | `StepScalars` |
| `SushiAI/optim/step_schedule.hpp` | `StepSchedule`, `validate_step_schedule` |
| `SushiAI/optim/step_schedule_section.hpp` | `K_STEP_SCHEDULE_TYPE`, `read_step_schedule_section`, `write_step_schedule_section` |

## Run state

`Optimizer::run_state()` lists what an optimizer keeps across updates and its options do not
state, so that a run can be saved and resumed. `type_name()` is the name a training file gives
the type: `K_SGD_TYPE`, "sgd", or `K_ADAMW_TYPE`, "adamw".

| Name | Kind | Kept by |
| --- | --- | --- |
| `step_count` | Scalar, a count | Every optimizer |
| `loss_scale` | Scalar, a real | An optimizer with a loss scaler attached |
| `loss_scale.good_steps` | Scalar, a count | An optimizer with a loss scaler attached |
| `<parameter>.momentum` | Tensor | `SGD` made with a momentum above 0 |
| `<parameter>.first_moment`, `<parameter>.second_moment` | Tensor | `AdamW` |

`<parameter>` is `ParameterSlot::name`, and the tensors come in slot order. A listed tensor
shares the optimizer's storage, so writing its elements restores it. The scratch tensors of
the separate kernels and the step scalars are not state: `write_step_scalars` recomputes the
scalars from the options and the step count before every update.

`restore_scalars(scalars)` takes exactly the scalars `run_state()` lists and sets them. It
returns `Incompatible`, naming the scalar and the optimizer type, for one that is missing,
surplus, stated twice or of the other kind, and `LossScaleState::restore` refuses a scale that
is no power of two within its bounds and a clean-step count at or above the growth interval.
`check_state_owner(owner)` refuses state another type wrote, with both type names.
`IO::save_run_checkpoint` and `IO::load_run_checkpoint` write and read the listing.
`tests/test_run_state.cpp` covers the names, the restore and each refusal.

## The step schedule

`StepSchedule::rate_at(iteration)` returns the learning rate of one update. Below
`warmup_iterations` it is `base_rate * (iteration / warmup_iterations) ^ warmup_power`; after
that it is `base_rate` times the scale of every step whose iteration has been reached.
`validate_step_schedule` refuses a schedule whose steps do not ascend or whose scales are not as
many as its steps. The schedule holds no state and names no optimizer. `tests/test_step_schedule.cpp` covers it.

## The schedule section

`read_step_schedule_section` and `write_step_schedule_section` convert a `StepSchedule` to and
from the `schedule` object of a training file. The object is a `Config::Value`, which is why
this module depends on `config`. Each key is the name of the field it states:

| Key | Holds | Field |
| --- | --- | --- |
| `type` | `"step"`, `K_STEP_SCHEDULE_TYPE` | none |
| `base_rate` | A finite number | `base_rate` |
| `warmup_iterations` | A whole number, 0 or more | `warmup_iterations` |
| `warmup_power` | A finite number, 0 or more | `warmup_power` |
| `steps` | A list of whole numbers, strictly ascending | `steps` |
| `scales` | A list of finite numbers, as long as `steps` | `scales` |

```json
"schedule": {
    "type": "step",
    "base_rate": 0.00261,
    "warmup_iterations": 1000,
    "warmup_power": 4.0,
    "steps": [400000, 450000],
    "scales": [0.1, 0.1]
}
```

The reader gives a key left out the default of its field, so `{}` reads as `StepSchedule{}`,
whose `base_rate` is 0. It accepts the object with `type` or with `type` taken out, as
`Config::TrainSpec` hands a typed block to its factory; a `type` that is stated must be
`"step"`. A key outside the table is refused with the keys the section accepts. The writer
states `type` and every key, in the order of the table.

Both functions refuse a rate, a power or a scale that is not finite, because JSON holds no such
number, and then run `validate_step_schedule`. Every refusal is `InvalidDescription` and starts
with `schedule`; one that concerns a key names it, as in
`schedule 'steps' entry 1 must be a whole number, but it is a number`. A refusal of
`validate_step_schedule` keeps that function's message behind the prefix `schedule: `.

A double is written as itself, so `read(write(schedule))` returns every field unchanged, and
`write(read(section))` returns a section that states every key in the order of the table. A
whole number where the table says number is read and written back as a real.

One table in `step_schedule_section.cpp` holds each key with its reader and its writer, so a
new field of `StepSchedule` is one row there. The private header `source/schedule_section.hpp`
holds what every schedule section shares: the row type and the loops that read and write a
table. The checks of the keys, of `type` and of each kind are those of
`Config::SectionReader`, so the texts of the refusals are the ones `core/config` states. `tests/test_step_schedule_section.cpp`
round-trips the defaults and a two-step schedule with a tolerance of zero and has a case for
each refusal.

## The cosine schedule

`CosineSchedule::rate_at(iteration)` is `base_rate * iteration / warmup_iterations` below
`warmup_iterations`, `final_rate` from `total_iterations` on, and between the two
`final_rate + (base_rate - final_rate) * (1 + cos(pi * progress)) / 2`, where `progress` is
`(iteration - warmup_iterations) / (total_iterations - warmup_iterations)`.
`validate_cosine_schedule` refuses a negative warm-up, a decay that ends before the warm-up
does, and a final rate that is negative or above the base rate. Its section states
`"type": "cosine"` and the four fields under their own names:

```json
"schedule": {
    "type": "cosine",
    "base_rate": 0.0006,
    "warmup_iterations": 2000,
    "total_iterations": 600000,
    "final_rate": 0.00006
}
```

`tests/test_cosine_schedule.cpp` covers the rate at the quarters of the decay, the check and
the section.

## A schedule of any type

`RateSchedule` is `std::variant<StepSchedule, CosineSchedule>`: the value a training file
gives when the type is read from the file. It is a variant and not an interface because a
schedule is a plain struct that a section reads and writes field by field, the set of types is
the closed set a file may name, and code that holds a `const StepSchedule*` keeps compiling.
`read_rate_schedule_section` reads `type` and calls the reader of that row of its table; a
section that states no `type` is read as `step`, and a type no row has is refused with every
type there is. `apply_step_schedule` is a template over any type with `rate_at`, and has one
overload over the variant. `tests/test_rate_schedule.cpp` covers the table.

## The rate of each update

An optimizer does not bake its rate into the compiled step. `SGD` and `AdamW` keep it in their
options and write it into a `StepScalars` buffer in `write_step_scalars`, which
`Optimizer::before_replay` calls before every replay, in the plan or not. A rate set between two
updates is therefore the rate of the next one, and the step is compiled once.

`Optimizer::set_learning_rate(rate)` sets it for any optimizer; a rate that is negative or not
finite stops the process. `apply_step_schedule(schedule, optimizer)` sets
`schedule.rate_at(optimizer.step_count() + 1)` and returns it. The iteration comes from the
optimizer's own count of applied updates, so the caller keeps no counter and an update the loss
scaler skipped does not use up an iteration. `tests/test_apply_step_schedule.cpp` covers both
optimizers and the parameter after three updates.

## One source of the rate

`read_rate_schedule(schedule, optimizer_attributes)` reads the schedule of a run, of either
type, from the two sections of a training file that could each state a rate.
It returns a `RateSchedule`, the variant of `StepSchedule` and `CosineSchedule`;
`apply_step_schedule` and `validate_rate_schedule` take the variant and act on the type held,
so a loop names neither type.

| The file states | Result |
| --- | --- |
| No `schedule` | No schedule; the optimizer trains at its `learning_rate`, or at its default |
| `schedule` with a `base_rate` above zero, no `optimizer.learning_rate` | The schedule |
| `schedule` and `optimizer.learning_rate` | `InvalidDescription`, even when the two are equal |
| `schedule` without `base_rate`, or with one at or below zero | `InvalidDescription` |

Neither section is the default of the other, so no file trains at a rate it does not state.
`tests/test_run_schedule.cpp` has a case for each row.

## Weight decay by slot

`ParameterSlot::decays` says whether weight decay applies to a parameter. A slot built by hand
decays; `Train::bind_parameters` sets the flag from the parameter's role. Both kernels of both
optimizers skip the decay of a slot that does not take it. The option `decay_all`, false by
default, decays every slot; a training file states it as `optimizer.decay_all`.
`tests/test_parameter_decay.cpp` covers both optimizers, both kernels and the key.

## Clipping by global norm

`Optimizer::enable_gradient_clip(max_norm)` builds a `GradientClip`, and the optimizer then
records, in front of its updates, one `nrm2` per gradient and one task that folds the norms
into a one-element factor: `max_norm / norm` when the norm over every gradient is above
`max_norm`, and 1 otherwise. Each fused update declares a read of that factor and multiplies
its gradient by it, so the device orders the three stages and the host reads nothing back.
`max_norm` is a step scalar: `set_clip_norm` sets the norm of the next update, and the step is
compiled once. Under loss scaling the norm is that of the unscaled gradient.

An optimizer without a clip records no task for it and its kernels are the ones it had. A
separate-kernel optimizer refuses a clip, and so do gradients of more than one element type.
A training file states the norm as `optimizer.clip_norm`; the key left out means no clip.
`GradientClip::applied_scale()` reads the factor of the last update. The folding task squares
each gradient's norm, so a norm above the square root of the largest float gives a factor of 0.
`tests/test_gradient_clip.cpp` covers both optimizers, the task count and the key.

The folding task is recorded as `sushiai.optim.clip_scale`. Its identity is
`Ops::K_CLIP_SCALE`, a row of `Ops::K_ALL` listed in `Graph::K_UNLOWERED` beside the two
update identities: no graph node carries it.
