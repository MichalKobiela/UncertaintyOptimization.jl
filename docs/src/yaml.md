```@meta
CurrentModule = UncertaintyOptimization
```

# YAML Model Files

YAML files define the symbolic model and parameter metadata. They do not define
the full run: observation times, initial conditions, solver choice, observed
states, and sampling settings are supplied later in Julia with `SimulationSpec`,
`TuringSpec`, and `CartesianScanner`.

A minimal example file is included in the repository at
`docs/src/examples/minimal_model.yml`.

```yaml
experiment:
  name: "MinimalDoseResponse"
  description: "One-state example showing fixed, uncertain, design, warmup, and staged parameters."

model:
  type: "ODE"
  states: ["X"]

equations:
  X: "drive * dose * input - decay * X + baseline"

parameters:
  drive:
    value: 1.0
    role: design
    bounds: [0.1, 5.0]
    design:
      warmup_value: 0.5
      value: 1.5
    design_optimise:
      scalers: "0.5:0.25:2.0"

  decay:
    value: 0.25
    role: uncertain
    prior:
      distribution: uniform
      lower: 0.05
      upper: 1.0

  dose:
    warmup_value: 0.0
    value: [0.5, 1.0, 2.0]
    role: fixed

  baseline:
    value: 0.1
    role: fixed

inputs:
  type: "step"
  t_threshold: 5.0
  values: [0.0, 1.0]
```

Load and compile it in the same way as any other model:

```julia
using ModelingToolkit
using ModelingToolkit: t_nounits as t
using UncertaintyOptimization

model_def = load_model("docs/src/examples/minimal_model.yml")
@mtkcompile sys = System(model_def.equations, t)
model = Model(model_def, sys)
```

## Top-Level Sections

`experiment` stores human-readable provenance. `name` and `description` are
copied into `ModelDefinition`.

`model` declares the model type and state names. ODE models are the active path
today. Every state listed here becomes a time-dependent ModelingToolkit
variable.

`equations` maps each state name to the right-hand side of its differential
equation. Equations are Julia expressions written as strings and may refer to
states, parameters, and the generated `input` signal.

`parameters` defines every parameter used by the equations. Each parameter has a
`role`, and may also define values, priors, bounds, warmup values, and
design-stage metadata.

`inputs` defines the input signal. The current loader supports a step input:
before `t_threshold` it uses `values[1]`, and after that threshold it uses
`values[2]`.

## Parameter Roles

`fixed` parameters are ordinary scalar or staged values. They are not sampled
during inference. In the example, `baseline` is scalar and `dose` is staged.

`uncertain` parameters are tunable parameters. They need an initial scalar
`value` and a `prior` when used with `TuringSpec`. In the example, `decay` is
sampled from a uniform prior.

`design` parameters are candidates for design-stage scans. They can have a
normal `value`, optional `bounds`, optional nested `design` values, and optional
`design_optimise` metadata. In the example, `drive` is a design parameter that
can be changed during Thompson sampling or evaluation.

## Parameter Fields

`value` is the ordinary value used for simulation and inference. A scalar value
is used for every solve. A vector value such as `[0.5, 1.0, 2.0]` is converted to
a tuple and creates staged production solves.

`warmup_value` is used for a warmup solve before the production solve. After
warmup, the final warmup state becomes the initial condition for each production
solve.

`prior` describes the distribution for an `uncertain` parameter. The currently
supported prior metadata is:

```yaml
prior:
  distribution: uniform
  lower: 0.05
  upper: 1.0
```

`bounds` records a design range. Bounds are metadata for design workflows; the
scanner receives its concrete grid through `CartesianScanner`.

`design` gives design-stage values for `run_scan`, which calls `simulate!` with
`design = true`. `design.warmup_value` is used during the design warmup solve,
and `design.value` is used for design-stage production solves.

`design_optimise.scalers` can store candidate scale factors in YAML. It accepts
either an explicit vector or `start:step:stop` notation:

```yaml
design_optimise:
  scalers: "0.5:0.25:2.0"
```

The scanner still receives its `scan = [...]` values in Julia. Use this YAML
metadata when you want to build that scan grid from the model definition.

## Warmup Solves

Warmup is useful when an experiment has a pre-stimulus or equilibration stage.
Any parameter with `warmup_value` is set to that value during warmup. Then
`simulate!` replaces the production initial state with the final warmup state.

In the minimal example, `dose` is `0.0` during warmup and then takes production
values `[0.5, 1.0, 2.0]`.

```yaml
dose:
  warmup_value: 0.0
  value: [0.5, 1.0, 2.0]
  role: fixed
```

To inspect the warmup solution, call:

```julia
sim = simulate!(
    model,
    simulation.initial_conditions,
    simulation.tspan;
    return_simulate = true,
)

sim.warmup_sol
sim.sols
```

`sim.warmup_sol` is `nothing` when no warmup values are configured.

## Multiple Experiments And Staged Parameters

Vector-valued parameter `value`s represent multiple production conditions. The
model is built once, then `simulate!` runs one production solve per vector
position.

```yaml
dose:
  warmup_value: 0.0
  value: [0.5, 1.0, 2.0]
  role: fixed
```

This creates three production solves:

1. `dose = 0.5`
2. `dose = 1.0`
3. `dose = 2.0`

If more than one parameter has a vector value, their vector lengths must match.
Values are paired by position. Scalar parameters are reused for every staged
solve.

During inference, each posterior proposal is evaluated against all staged
production solves. The observed data is flattened internally and must match:

```julia
length(simulation.t_obs) *
observed_state_count(simulation) *
number_of_staged_production_solves
```

For the minimal example, one observed state, ten observation times, and three
staged `dose` values require `30` observations. The order is stage first, then
observed state, then time. For one state this means all time points for
`dose = 0.5`, then all time points for `dose = 1.0`, then all time points for
`dose = 2.0`.

## Design Stage Values

Design-stage scans use the same model but may need different values from the
ordinary inference stage. Put those under a nested `design` block.

```yaml
drive:
  value: 1.0
  role: design
  bounds: [0.1, 5.0]
  design:
    warmup_value: 0.5
    value: 1.5
```

When `run_scan` evaluates a candidate, the default evaluator calls `simulate!`
with `design = true`. This means design warmup values and design production
values are used where they are defined, while uncertain parameters still come
from the posterior sample.
