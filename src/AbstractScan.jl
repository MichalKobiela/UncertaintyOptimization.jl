"""
    ThompsonSpec

Base type for scan specifications.
"""
abstract type ThompsonSpec end


"""
    ThompsonGridSpec(; simulation, scan, loss)

Grid-scan specification using a shared `SimulationSpec`.
"""
struct ThompsonGridSpec <: ThompsonSpec
    simulation::SimulationSpec
    scan::Vector{NamedTuple{(:symbol, :values, :kind), Tuple{Symbol, Vector{Float64}, Symbol}}}
    combinations::Vector{Vector{Float64}}
    loss::Any

    function ThompsonGridSpec(;
        simulation::SimulationSpec,
        scan,
        loss,
    )
        normalized_scan = collect(map(scan) do axis
            symbol = Symbol(axis.symbol)
            values = Float64.(collect(axis.values))
            kind = Symbol(axis.kind)

            if isempty(values)
                error("CartesianSampler.scan values for $symbol must not be empty")
            end

            if !all(isfinite, values)
                error("CartesianSampler.scan values for $symbol must contain finite values")
            end

            if !(kind in (:scale, :value))
                error("CartesianSampler.scan kind for $symbol must be either :scale or :value. Got :$kind.")
            end

            return (symbol = symbol, values = values, kind = kind)
        end)

        if isempty(normalized_scan)
            error("CartesianSampler.scan must not be empty")
        end

        return new(simulation, normalized_scan, _scan_combinations(normalized_scan), loss)
    end
end

"""
    CartesianSampler

Alias for `ThompsonGridSpec`.
"""
const CartesianSampler = ThompsonGridSpec

"""
    _scan_combinations(scan)

Build the Cartesian product of scan axis values.
"""
function _scan_combinations(scan)
    combinations = [Float64[]]

    for axis in scan
        next_combinations = Vector{Vector{Float64}}(undef, length(combinations) * length(axis.values))
        index = 1

        for combination in combinations, value in axis.values
            next = Vector{Float64}(undef, length(combination) + 1)
            copyto!(next, combination)
            next[end] = value
            next_combinations[index] = next
            index += 1
        end

        combinations = next_combinations
    end

    return combinations
end
