```@meta
CurrentModule = UncertaintyOptimization
```

# Reference

The API below is ordered in the same chronology as a typical analysis script:
load the YAML model, compile it with ModelingToolkit, wrap it in a runtime
`Model`, define reusable simulation settings, run direct simulation or
inference, and then scan design candidates.

```@index
Pages = ["reference.md"]
```

## Load A YAML Model

Start by loading the mechanistic model definition from YAML. The returned
`ModelDefinition` is declarative: it contains symbolic equations, states,
parameters, inputs, and metadata, but it is not yet an executable ODE problem.

```@docs
load_model
ModelDefinition
```

## Compile A ModelingToolkit System

Compile the equations from the loaded model definition with ModelingToolkit's
`System` constructor. This is the bridge between the package's YAML layer and
the SciML simulation layer.

```julia
using ModelingToolkit
using ModelingToolkit: t_nounits as t

model_def = load_model("model.yml")
@mtkcompile sys = System(model_def.equations, t)
```

`System` is provided by ModelingToolkit rather than
UncertaintyOptimization.jl, so its full constructor API is documented upstream.
Within this package workflow, pass the compiled `sys` directly to `Model`.

## Create The Runtime Model

Wrap the loaded definition and compiled system in `Model`. This object owns the
cached ODE problem and parameter setters reused by simulation, inference, and
scan calls.

```@docs
Model
```

## Define Simulation Settings

Use `SimulationSpec` for observation times, observed state or states, initial
conditions, time spans, solver choice, and solver options. The same spec is
shared by direct simulation, `TuringSpec`, and `CartesianScanner`.

```@docs
SimulationSpec
```

## Simulate

`simulate!` is the direct simulation entry point. It also underpins inference
and scan stages, which call it repeatedly after the model has been prepared.

```@docs
simulate!
```

## Infer Uncertain Parameters

Create a `TuringSpec` from a `SimulationSpec` and observed data, then pass it to
`run_inference`. The resulting posterior samples can be passed to `run_scan`.

```@docs
TuringSpec
run_inference
```

## Scan Design Candidates

Use `CartesianScanner` to define a Cartesian grid over design candidates and a
loss function. `run_scan` evaluates that grid for each posterior sample. The
same API is used for Thompson sampling and for later evaluation of selected
candidates.

```@docs
CartesianScanner
ThompsonGridSpec
run_scan
```

## Lower-Level Utilities

Most users do not need these directly, but they are exported or documented for
advanced workflows that prepare models explicitly.

```@docs
setup_simulation!
```
