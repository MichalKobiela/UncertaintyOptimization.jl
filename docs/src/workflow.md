```@meta
CurrentModule = UncertaintyOptimization
```

# Workflow

UncertaintyOptimization separates a study into a few reusable building blocks:

1. YAML model definition.
2. Simulation/build stage.
3. Inference.
4. Thompson sampling.
5. Evaluation.

The same loaded model and `SimulationSpec` can be reused across these stages so
that expensive ModelingToolkit and SciML setup is paid once and then reused.

## YAML Model Definition

The YAML file describes the symbolic model, not the full experiment execution.
`load_model` reads this file and returns a `ModelDefinition`. See
[YAML Model Files](@ref) for a complete minimal file and details about
parameter roles, warmup values, and staged parameters.

Required model sections are:

- `experiment`: a name and description for provenance.
- `model`: model type and state names.
- `parameters`: parameter values, roles, priors, and optional design metadata.
- `equations`: one right-hand-side expression for each declared state.
- `inputs`: currently a step input signal.

Parameter roles control how later stages treat each parameter:

- `fixed`: scalar value used directly in simulation.
- `uncertain`: tunable parameter with a prior for inference.
- `design`: parameter that may be changed during Thompson sampling or
  evaluation.

```julia
using ModelingToolkit
using ModelingToolkit: t_nounits as t
using UncertaintyOptimization

model_def = load_model("model.yml")
@mtkcompile sys = System(model_def.equations, t)
model = Model(model_def, sys)
```

Simulation settings such as observation times, observed states, solver choice,
initial conditions, and time span are supplied with `SimulationSpec` in Julia
code. This keeps the model definition separate from the execution stage.

## Simulation And Build Stage

`SimulationSpec` describes how a model should be solved for inference, scanning,
or direct simulation. `simulate!` is the main simulation entry point. On its
first call it delegates to `setup_simulation!`, which builds and caches the
`ODEProblem` and the parameter setters used by later calls.

```julia
using OrdinaryDiffEq

simulation = SimulationSpec(
    t_obs = collect(range(1.0, 90.0, length = 30)),
    obs_state = :A,
    initial_conditions = (1.0, 1.0),
    tspan = (0.0, 100.0),
    solver = Rosenbrock23(),
    solver_opts = (dtmin = 1e-9,),
)

sols = simulate!(model, simulation)
```

When warmup and production need different time spans, pass a pair of intervals:

```julia
simulation = SimulationSpec(
    t_obs = collect(range(0.0, 5.45, length = 30)),
    obs_state = :A,
    initial_conditions = (1.0, 1.0),
    tspan = ((0.0, 10.0), (0.0, 5.45)),
    solver = Rosenbrock23(),
)
```

Warmup values and tuple-valued staged parameters are taken from the YAML
parameter metadata. With `return_simulate = true`, `simulate!` returns both the
optional warmup solution and the production solutions.

## Inference

Inference estimates uncertain parameters from observations. For Turing-based
Bayesian inference, create a `TuringSpec` from a `SimulationSpec` and observed
data, then call `run_inference`.

```julia
using Distributions
using Turing

turing_spec = TuringSpec(
    simulation = simulation,
    data = observed_data,
    noise_prior = InverseGamma(2, 3),
    noise_initial = 3.0,
    sampler = NUTS(0.65),
    n_samples = 3000,
    n_chains = 1,
)

chain = run_inference(model, turing_spec)
```

`run_inference` prepares the model for simulation, builds priors for parameters
marked `uncertain` in YAML, calls `simulate!` inside the Turing model, and
returns a chain with parameter names restored to the YAML symbols.

## Thompson Sampling

Thompson sampling evaluates a grid of candidate design choices for each
posterior draw. `CartesianScanner` is an alias for `ThompsonGridSpec`; it defines
the shared simulation settings, the candidate grid, and a scalar loss function.

```julia
function loss(warmup_sol, predicted_sol; sys = nothing)
    target = 50.0
    prediction = Array(predicted_sol)[end]
    return (prediction - target)^2
end

scan = CartesianScanner(
    simulation = simulation,
    scan = [
        (symbol = :kx2, values = LinRange(0.01, 3.0, 100), kind = :scale),
    ],
    loss = loss,
)

thompson_results = run_scan(chain, scan, model)
chosen_designs = thompson_results[thompson_results.is_best, :]
```

For `kind = :scale`, each grid value multiplies the posterior draw for an
uncertain parameter, or the scalar YAML value for a fixed/design parameter. For
`kind = :value`, the grid value is used directly.

## Evaluation

Evaluation reuses `run_scan`, usually over a smaller set of candidate designs
chosen during Thompson sampling. The returned table has one row per posterior
iteration and candidate, including the candidate values, the resolved parameter
values, `loss`, `is_best`, and `best_loss`.

```julia
candidate_values = unique(chosen_designs.kx2_scaler)

evaluation_scan = CartesianScanner(
    simulation = simulation,
    scan = [
        (symbol = :kx2, values = candidate_values, kind = :scale),
    ],
    loss = loss,
)

evaluation_results = run_scan(chain, evaluation_scan, model)
```

Common evaluation summaries group by the candidate columns and aggregate losses,
for example median loss, upper quantiles, standard deviation, or the number of
times a candidate was selected during Thompson sampling.
