"""
    ThompsonSpec

Base type for Thompson-style scan specifications.

Scan specs describe a candidate design space and a loss function. `run_scan`
evaluates that design space for each posterior sample, which supports both the
Thompson sampling stage and the later evaluation stage.
"""
abstract type ThompsonSpec end


"""
    ThompsonGridSpec(; simulation, scan, loss)

Grid-scan specification using a shared `SimulationSpec`.

`ThompsonGridSpec` defines the design candidates considered for each posterior
sample. `scan` is a collection of named tuples with:

- `symbol`: parameter to modify.
- `values`: candidate values or scale factors.
- `kind`: either `:scale` or `:value`.

For `kind = :scale`, each grid value multiplies the posterior draw when the
symbol is uncertain, or the scalar YAML value when the symbol is fixed/design.
For `kind = :value`, the grid value is used directly as the parameter value.

`loss` receives the evaluated simulation output and returns a scalar. The
default evaluator calls:

```julia
loss(warmup_sol, predicted_sol; sys=model.sys)
```

The constructor validates scan axes and precomputes the Cartesian product in
`combinations`.
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
                error("CartesianScanner.scan values for $symbol must not be empty")
            end

            if !all(isfinite, values)
                error("CartesianScanner.scan values for $symbol must contain finite values")
            end

            if !(kind in (:scale, :value))
                error("CartesianScanner.scan kind for $symbol must be either :scale or :value. Got :$kind.")
            end

            return (symbol = symbol, values = values, kind = kind)
        end)

        if isempty(normalized_scan)
            error("CartesianScanner.scan must not be empty")
        end

        return new(simulation, normalized_scan, _scan_combinations(normalized_scan), loss)
    end
end

"""
    CartesianScanner

Alias for `ThompsonGridSpec`.

Use this name when the important detail is the Cartesian grid search over design
candidates. Use `ThompsonGridSpec` when emphasizing the Thompson sampling stage
that evaluates the grid under posterior draws.
"""
const CartesianScanner = ThompsonGridSpec

"""
    CartesianSampler

Backward-compatible alias for `CartesianScanner`.
"""
const CartesianSampler = CartesianScanner

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
