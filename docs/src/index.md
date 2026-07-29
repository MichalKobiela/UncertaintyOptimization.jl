```@meta
CurrentModule = UncertaintyOptimization
```

# UncertaintyOptimization

Documentation for [UncertaintyOptimization](https://github.com/MichalKobiela/UncertaintyOptimization.jl).

Package for risk-averse optimization under uncertainty.

## Publication

This package accompanies the workflow described in [Risk-averse optimization of genetic circuits under uncertainty](https://www.cell.com/cell-systems/fulltext/S2405-4712(25)00309-6) by **Michal Kobiela**, **Diego A. Oyarzun**, and **Michael U. Gutmann**, published in *Cell Systems*. The paper presents a design strategy for genetic circuits whose mechanistic models contain **uncertain parameters** and **design parameters**. Observed data from previous designs, including non-functional prototypes, are used to infer a posterior distribution over uncertain parameters. Candidate design parameters are then optimized through Thompson sampling and ranked with risk-averse summaries of predictive loss, so final designs are chosen for performance under epistemic uncertainty (posterior uncertainty) and any additional stochasticity you include in the model or loss.

## Install

```julia-repl
pkg> add https://github.com/MichalKobiela/UncertaintyOptimization.jl#1.0.0
```

## ModelingToolkit Integration

The package uses [ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/),
or MTK, as the bridge between a model description and numerical simulation in
Julia. YAML files describe states, equations, and parameter metadata. The
package converts those declarations into symbolic ModelingToolkit equations,
which are then compiled into a system and wrapped in a runtime `Model`.

MTK provides several important advantages for this workflow:

- **Symbolic model construction**: equations can be manipulated and inspected
  before numerical simulation.
- **Consistent parameter and state handling**: symbolic names provide a stable
  interface for fixed, uncertain, and design parameters.
- **SciML integration**: compiled systems can be used to construct
  `ODEProblem`s and solve them with the Julia differential-equation ecosystem.
- **Analytic Jacobians**: MTK can derive Jacobians symbolically and provide them
  to numerical solvers, avoiding finite-difference approximations and often
  improving the efficiency and robustness of stiff ODE solves.
- **Reusable numerical setup**: the compiled system, problem structure, and
  parameter setters can be prepared once and reused across many simulations.
- **A path to model transformations**: the same symbolic representation can
  support simplification, automatic equation transformations, sensitivity
  analysis, and extensions to larger mechanistic models.

In Julia, MTK is part of the SciML ecosystem: it handles the symbolic layer,
while packages such as OrdinaryDiffEq provide numerical solvers and Turing
provides probabilistic inference. UncertaintyOptimization.jl connects these
layers. `load_model` parses the YAML model, `Model` holds the compiled MTK
system and reusable simulation state, `simulate!` solves the resulting model,
and the inference and scan stages repeatedly evaluate it for posterior and
design parameter values.

The package workflow uses these stages:

- **Model the system**: YAML model files and `load_model` define the mechanistic model, states, equations, fixed parameters, uncertain parameters, and design parameters.
- **Identify uncertain and design parameters**: YAML parameter `role`s separate uncertain parameters to infer from controllable design parameters to optimize.
- **Infer uncertain parameters**: `TuringSpec` and `run_inference` use observed data to produce posterior samples for parameters marked `uncertain`.
- **Define the design goal**: the `loss` function supplied to `CartesianScanner` encodes the desired circuit behavior, such as matching a set point, trajectory, amplitude, or frequency.
- **Optimize designs via Thompson sampling**: `CartesianScanner` and `run_scan` evaluate candidate design values for each posterior draw. Each best candidate for a posterior draw is a grid-based Thompson sample.
- **Manage risk and select final designs**: the `run_scan` output records candidate values, resolved parameters, losses, and best-design markers so downstream summaries can rank designs by median loss, upper quantiles, or other risk-averse criteria.

## Reference Code

The repository also keeps reference code for comparing the package workflow against earlier scripts. For this code, the docs use a one-page starter style: a single script that shows the full setup, mechanistic model, observed data, inference call, and saved output in one place. The current starter is [`experiments/reference/reproduce_original_code.jl`](https://github.com/MichalKobiela/UncertaintyOptimization.jl/blob/main/experiments/reference/reproduce_original_code.jl), which reproduces the adaptation-circuit inference workflow with newer Julia libraries and colocated reference data.

The package workflow is organized around a small set of stages:

- load a symbolic model from YAML;
- build and reuse simulations with `SimulationSpec` and `simulate!`;
- infer uncertain parameters from observed data with `TuringSpec` and `run_inference`;
- obtain Thompson samples by scanning design candidates with `CartesianScanner` and `run_scan`;
- evaluate candidate designs and summarize predictive loss for risk-averse selection.

See [YAML Model Files](@ref) for model-file structure, [Workflow](@ref) for the
stage-by-stage guide, and [Reference](@ref) for generated API documentation.

## Implementation

The original code developed by **Michal Kobiela** was transformed and extended
by **Mateusz Bieniek** and **Emma Pead** as part of the [**Data Intelligence
Hub**](https://informatics.ed.ac.uk/ukri-cdt-in-biomedical-ai/data-intelligence-hub)
in the School of Informatics at the University of Edinburgh. This work
produced a more structured and reusable Julia package that integrates
ModelingToolkit (MTK) with
the SciML ecosystem, separating symbolic model construction from simulation,
Bayesian inference, design scanning, and risk evaluation.

Model definitions are expressed in YAML. This gives parameters an explicit
structure, including their values, roles (`fixed`, `uncertain`, or `design`),
priors, bounds, and staged warmup or production values. Equations are also
declared in YAML and converted into symbolic MTK equations before the numerical
problem is constructed. This makes models easier to inspect, modify, validate,
and reuse across experiments.

The codebase also includes a testing framework covering model loading,
simulation, inference, scanning, staged parameters, and YAML validation. These
tests, together with clearer interfaces, documented workflows, validation, and
reusable simulation setup, improve the overall quality, maintainability, and
reproducibility of the implementation.

Performance has been improved by reusing compiled MTK systems and ODE problems,
precomputing parameter setters, using analytic Jacobians where available, and
reducing repeated setup work in inference and design scans. Bayesian inference
has also been tuned for performance, including configuration and optimisation
of the NUTS sampler for the model and data, while retaining the flexibility to
adjust sampling settings for different experiments.
