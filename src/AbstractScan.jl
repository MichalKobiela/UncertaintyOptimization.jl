"""
    GridScan

Specification for grid-scan style inference/optimisation.

Uses a shared `SimulationSpec` for simulation settings, plus:

- `scale`: parameter to scan/scale, as a `Symbol` or `String`
- `values`: scalers to scan over
- `lossf`: user-provided loss function used to score a scan candidate

Example
```julia
spec = GridScan(
    simulation = sim_spec,
    scale = "kx2",
    values = LinRange(0.01, 3, 100),
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
    lossf::Any

    function GridScan(;
        simulation::SimulationSpec,
        scale::Union{Nothing, Symbol, AbstractString} = nothing,
        values = Float64[],
        lossf,
    )
        grid_values = Float64.(collect(values))
        if !isempty(grid_values) && !all(isfinite, grid_values)
            error("GridScan.values must contain finite values")
        end

        scale_symbol = isnothing(scale) ? nothing : Symbol(scale)

        return new(
            simulation,
            scale_symbol,
            grid_values,
            lossf,
        )
    end
end
