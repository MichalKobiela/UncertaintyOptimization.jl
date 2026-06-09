"""
    GridScan

Specification for grid-scan style inference/optimisation.

Uses a shared `SimulationSpec` for simulation settings, plus:

- `scale`: parameter to scan/scale, as a `Symbol` or `String`
- `values`: scalers to scan over
- `kind`: `:scale` multiplies `scale` by each value, `:value` sets `scale` to each value
- `lossf`: user-provided loss function used to score a scan candidate

Example
```julia
spec = GridScan(
    simulation = sim_spec,
    scale = "kx2",
    values = LinRange(0.01, 3, 100),
    kind = :scale,
    lossf = loss,
)
```
"""


# this a spec that defines how to apply a loss function to the posterior
abstract type ScanSpec end



struct GridScan <: ScanSpec
    # simulation
    simulation::SimulationSpec

    # actual grid scan params
    scale::Union{Nothing, Symbol}
    values::Vector{Float64}
    kind::Symbol
    lossf::Any

    function GridScan(;
        simulation::SimulationSpec,
        scale::Union{Nothing, Symbol, AbstractString} = nothing,
        values = Float64[],
        kind::Union{Symbol, AbstractString} = :scale,
        lossf,
    )
        grid_values = Float64.(collect(values))
        if !isempty(grid_values) && !all(isfinite, grid_values)
            error("GridScan.values must contain finite values")
        end

        scale_symbol = isnothing(scale) ? nothing : Symbol(scale)
        kind_symbol = Symbol(kind)
        if !(kind_symbol in (:scale, :value))
            error("GridScan.kind must be either :scale or :value. Got :$kind_symbol.")
        end

        return new(
            simulation,
            scale_symbol,
            grid_values,
            kind_symbol,
            lossf,
        )
    end
end
