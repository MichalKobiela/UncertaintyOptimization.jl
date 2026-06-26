```@meta
CurrentModule = UncertaintyOptimization
```

# UncertaintyOptimization

Documentation for [UncertaintyOptimization](https://github.com/MichalKobiela/UncertaintyOptimization.jl).

Package for risk-averse optimisation under uncertainty.

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
