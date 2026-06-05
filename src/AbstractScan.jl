"""
    GridScan

Specification for grid-scan style inference/optimisation.

Uses a shared `SimulationSpec` for simulation settings, plus:

- `scale`: parameter to scan/scale, as a `Symbol` or `String`
- `linrange`: scalers to scan over
- `lossf`: user-provided loss function used to score a scan candidate

Example
```julia
spec = GridScan(
    simulation = sim_spec,
    scale = "kx2",
    linrange = LinRange(0.01, 3, 100),
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
    linrange::Vector{Float64}
    lossf::Any

    function GridScan(;
        simulation::SimulationSpec,
        scale::Union{Nothing, Symbol, AbstractString} = nothing,
        linrange::AbstractVector{<:Number} = Float64[],
        lossf,
    )
        linrange_values = Float64.(collect(linrange))
        if !isempty(linrange_values) && !all(isfinite, linrange_values)
            error("linrange values must be finite")
        end

        scale_symbol = isnothing(scale) ? nothing : Symbol(scale)

        return new(
            simulation,
            scale_symbol,
            linrange_values,
            lossf,
        )
    end
end

