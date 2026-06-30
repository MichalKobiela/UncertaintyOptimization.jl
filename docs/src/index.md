```@meta
CurrentModule = UncertaintyOptimization
```

# UncertaintyOptimization

Documentation for [UncertaintyOptimization](https://github.com/MichalKobiela/UncertaintyOptimization.jl).

Package for risk-averse optimisation under uncertainty.

## Publication

This package accompanies the workflow described in [Risk-averse optimization of genetic circuits under uncertainty](https://www.cell.com/cell-systems/fulltext/S2405-4712(25)00309-6) by Kobiela, Oyarzun, and Gutmann, published in *Cell Systems*. The paper presents a design strategy for genetic circuits whose mathematical models contain uncertain parameters. It uses data from previous designs, including non-functional prototypes, to infer a posterior distribution over uncertain model parameters, then applies Thompson sampling and risk-aware evaluation to choose designs that are expected to perform well despite epistemic uncertainty and biomolecular noise. The examples include adaptation circuits and genetic oscillators, showing how posterior uncertainty can be turned into more robust design choices.

The package building blocks correspond to the publication workflow as follows:

- **Model the system**: YAML model files and `load_model` define the mechanistic model, states, equations, fixed parameters, uncertain parameters, and design parameters. This corresponds to the paper's split of model parameters into uncertain parameters and controllable design parameters.
- **Compile and simulate**: `ModelDefinition`, `Model`, `SimulationSpec`, `setup_simulation!`, and `simulate!` provide the reusable ODE simulation layer used by inference, design scanning, and evaluation.
- **Infer uncertain parameters**: `TuringSpec` and `run_inference` implement the Bayesian inference stage for parameters marked `uncertain`, producing posterior samples analogous to the paper's inferred parameter distribution.
- **Define design goals**: the `loss` function supplied to `CartesianScanner` encodes the desired behavior, such as matching a set point, trajectory, amplitude, or frequency.
- **Optimize designs with Thompson samples**: `CartesianScanner` and `run_scan` evaluate candidate design values for each posterior draw, matching the paper's Thompson-sampling idea in a grid-scan form.
- **Evaluate and select robust designs**: the `run_scan` output records candidate values, resolved parameter values, losses, and best-design markers so downstream summaries can rank candidates by median loss, high quantiles, or other risk-aware criteria.

The package workflow is organised around a small set of stages:

- load a symbolic model from YAML;
- build and reuse simulations with `SimulationSpec` and `simulate!`;
- infer uncertain parameters with `TuringSpec` and `run_inference`;
- scan design candidates with `CartesianScanner` and `run_scan`;
- evaluate selected candidates by reusing the same scan machinery.

See [YAML Model Files](@ref) for model-file structure, [Workflow](@ref) for the
stage-by-stage guide, and [Reference](@ref) for generated API documentation.

## Install

```julia-repl
pkg> add https://github.com/MichalKobiela/UncertaintyOptimization.jl#v1.0.0
```
