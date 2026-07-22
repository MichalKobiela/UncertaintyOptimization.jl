```@meta
CurrentModule = UncertaintyOptimization
```

# UncertaintyOptimization

Documentation for [UncertaintyOptimization](https://github.com/MichalKobiela/UncertaintyOptimization.jl).

Package for risk-averse optimization under uncertainty.

## Publication

This package accompanies the workflow described in [Risk-averse optimization of genetic circuits under uncertainty](https://www.cell.com/cell-systems/fulltext/S2405-4712(25)00309-6) by **Michal Kobiela**, **Diego A. Oyarzun**, and **Michael U. Gutmann**, published in *Cell Systems*. The paper presents a design strategy for genetic circuits whose mechanistic models contain **uncertain parameters** and **design parameters**. Observed data from previous designs, including non-functional prototypes, are used to infer a posterior distribution over uncertain parameters. Candidate design parameters are then optimized through Thompson sampling and ranked with risk-averse summaries of predictive loss, so final designs are chosen for performance under epistemic uncertainty (posterior uncertainty) and any additional stochasticity you include in the model or loss.

The package workflow uses these stages:

- **Model the system**: YAML model files and `load_model` define the mechanistic model, states, equations, fixed parameters, uncertain parameters, and design parameters.
- **Identify uncertain and design parameters**: YAML parameter `role`s separate uncertain parameters to infer from controllable design parameters to optimize.
- **Infer uncertain parameters**: `TuringSpec` and `run_inference` use observed data to produce posterior samples for parameters marked `uncertain`.
- **Define the design goal**: the `loss` function supplied to `CartesianScanner` encodes the desired circuit behavior, such as matching a set point, trajectory, amplitude, or frequency.
- **Optimize designs via Thompson sampling**: `CartesianScanner` and `run_scan` evaluate candidate design values for each posterior draw. Each best candidate for a posterior draw is a grid-based Thompson sample.
- **Manage risk and select final designs**: the `run_scan` output records candidate values, resolved parameters, losses, and best-design markers so downstream summaries can rank designs by median loss, upper quantiles, or other risk-averse criteria.

## Reference Implementations

The repository also keeps reference implementations for comparing the package workflow against earlier scripts. For these, the docs use a one-page starter style: a single script that shows the full setup, mechanistic model, observed data, inference call, and saved output in one place. The current starter is [`experiments/reference/reproduce_original_code.jl`](https://github.com/MichalKobiela/UncertaintyOptimization.jl/blob/main/experiments/reference/reproduce_original_code.jl), which reproduces the adaptation-circuit inference workflow with newer Julia libraries and colocated reference data.

The package workflow is organized around a small set of stages:

- load a symbolic model from YAML;
- build and reuse simulations with `SimulationSpec` and `simulate!`;
- infer uncertain parameters from observed data with `TuringSpec` and `run_inference`;
- obtain Thompson samples by scanning design candidates with `CartesianScanner` and `run_scan`;
- evaluate candidate designs and summarize predictive loss for risk-averse selection.

See [YAML Model Files](@ref) for model-file structure, [Workflow](@ref) for the
stage-by-stage guide, and [Reference](@ref) for generated API documentation.

## Install

```julia-repl
pkg> add https://github.com/MichalKobiela/UncertaintyOptimization.jl#v1.0.0
```
