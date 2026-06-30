```@meta
CurrentModule = UncertaintyOptimization
```

# Workflow

UncertaintyOptimization follows the publication workflow for risk-averse
optimization under uncertainty. A study is organized around mechanistic model
definition, Bayesian inference for uncertain parameters, Thompson-sampling-based
design optimization, and risk-averse selection of final designs.

The publication-facing stages are:

1. Model the system.
2. Define the design goal.
3. Identify uncertain parameters and design parameters.
4. Infer uncertain parameters from observed data.
5. Optimize designs via Thompson sampling.
6. Manage risk and select final design(s).

The package adds a reusable simulation/build layer around those stages. The same
loaded model and `SimulationSpec` can be reused so expensive ModelingToolkit and
SciML setup is paid once and then reused by inference, Thompson sampling, and
evaluation.

## Model The System

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

Parameter roles encode the publication's parameter split:

- `fixed`: a high-confidence constant or staged experimental condition.
- `uncertain`: a parameter inferred from observed data through its prior and
  posterior distribution.
- `design`: a controllable parameter that may be changed during design
  optimization and risk-averse evaluation.

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
code. This keeps the mechanistic model separate from the observed-data and
design-optimization stages.

## Compile And Simulate

`SimulationSpec` describes how a model should be solved for inference, Thompson
sampling, risk-averse evaluation, or direct simulation. `simulate!` is the main
simulation entry point. On its first call it delegates to `setup_simulation!`,
which builds and caches the `ODEProblem` and the parameter setters used by later
calls.

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

These simulated trajectories are the model predictions compared with observed
data during inference and with target behavior during design evaluation.

## Infer Uncertain Parameters From Observed Data

Inference estimates uncertain parameters from observed data. In the publication,
this is the stage that turns prior knowledge into a posterior distribution over
uncertain parameters. In the package, parameters marked `uncertain` in YAML
receive priors, and `run_inference` samples their posterior with Turing.

For Turing-based Bayesian inference, create a `TuringSpec` from a
`SimulationSpec` and observed data, then call `run_inference`.

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

Each row of the returned chain is one posterior sample of the uncertain
parameters. Those posterior samples feed directly into the Thompson-sampling
stage.

## Define The Design Goal

The publication expresses the desired behavior as a loss function: low loss
means the simulated circuit is close to the target set point, trajectory,
amplitude, frequency, or other design objective. In the package, the `loss`
function supplied to `CartesianScanner` plays the same role.

```julia
function loss(warmup_sol, predicted_sol; sys = nothing)
    target = 50.0
    prediction = Array(predicted_sol)[end]
    return (prediction - target)^2
end
```

The default scanner evaluator passes the warmup solution and production solution
to this loss. If the design goal needs a different summary, provide a custom
loss, or provide a custom evaluator to `run_scan`.

## Optimize Designs Via Thompson Sampling

The publication obtains Thompson samples by drawing uncertain parameters from
the posterior and optimizing the design parameters for each draw. In this
package, `CartesianScanner` implements that idea with a grid of candidate design
values. For each posterior sample, `run_scan` simulates every candidate and marks
the lowest-loss candidate as the best design for that posterior draw.

`CartesianScanner` is an alias for `ThompsonGridSpec`; it defines the shared
simulation settings, candidate design grid, and scalar predictive-loss function.

```julia
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

The rows with `is_best == true` are the package's grid-based Thompson samples:
one selected design for each posterior draw.

## Manage Risk And Select Final Designs

The publication evaluates designs through the predictive loss distribution under
posterior uncertainty and biomolecular noise, then uses risk-averse summaries
such as upper quantiles to avoid designs that only work in favorable
conditions.

In the package, evaluation reuses `run_scan`, usually over a smaller set of
candidate designs chosen during Thompson sampling. The returned table has one
row per posterior sample and candidate, including candidate values, resolved
parameter values, `loss`, `is_best`, and `best_loss`.

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
for example median loss, upper quantiles such as the 75% quantile, standard
deviation, or the number of times a candidate was selected during Thompson
sampling. These summaries correspond to the publication's risk-neutral and
risk-averse views of design performance.

## From Package Outputs To Publication Quantities

The objects returned by the package map directly onto the quantities discussed
in the publication:

- `chain`: posterior samples of uncertain parameters inferred from observed
  data.
- `chosen_designs`: Thompson samples, i.e. best candidate designs under
  individual posterior samples.
- `evaluation_results.loss`: samples from each candidate design's predictive
  loss distribution.
- grouped summaries of `evaluation_results`: risk-neutral and risk-averse
  design rankings.

For example, the median loss approximates typical predictive performance, while
an upper quantile such as the 75% quantile is a risk-averse summary:

```julia
using DataFrames
using Statistics

risk_summary = combine(
    groupby(evaluation_results, :kx2_scaler),
    :loss => median => :median_loss,
    :loss => (x -> quantile(x, 0.75)) => :q75_loss,
)
```

Candidates with low median loss and low upper-quantile loss correspond to the
publication's robust designs: they are predicted to work well on average and to
avoid large losses under unfavorable posterior draws.
