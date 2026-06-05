using Tables

"""
    run_scan(samples, spec::GridScan; evaluator = nothing)

Run a one-dimensional grid scan for each sample in `samples` using `spec.linrange`.

By default, each loss is evaluated as:

```julia
spec.lossf(sample, grid_value)
```

If `evaluator` is provided, it is called first as:

```julia
evaluated = evaluator(sample, grid_value, spec)
```

Then the loss is evaluated as:

- `spec.lossf(evaluated...)` if `evaluated` is a tuple
- `spec.lossf(evaluated)` otherwise

This lets a caller provide an evaluator that turns `(posterior_sample, grid_value, spec)`
into simulation outputs such as `(warmup_sol, predicted_sol)`.

Returns a vector of named tuples with the best grid value and all losses for each sample.
"""
function run_scan(samples, spec::GridScan, model::Model; evaluator = nothing)
    if isempty(spec.linrange)
        error("GridScan.linrange must not be empty")
    end

    posterior_samples = _scan_rows(samples)
    results = NamedTuple[]

    for (sample_index, one_posterior) in enumerate(posterior_samples)
        losses = Vector{Float64}(undef, length(spec.linrange))

        @show one_posterior

        for (grid_index, grid_value) in pairs(spec.linrange)
            @show grid_value
            # use the simulate function to evaluate 
            # first, you have to scale the value
            # simulate!()

            loss_value = if isnothing(evaluator)
                spec.lossf(one_posterior, grid_value)
            else
                evaluated = evaluator(one_posterior, grid_value, spec)
                evaluated isa Tuple ? spec.lossf(evaluated...) : spec.lossf(evaluated)
            end

            if !(loss_value isa Real)
                error("GridScan.lossf must return a real scalar. Got $(typeof(loss_value)).")
            end

            losses[grid_index] = Float64(loss_value)
        end

        best_index = argmin(losses)
        push!(results, (
            sample_index = sample_index,
            sample = one_posterior,
            scale = spec.scale,
            best_value = spec.linrange[best_index],
            best_loss = losses[best_index],
            losses = losses,
        ))
    end

    return results
end

function _scan_rows(samples)
    if Tables.istable(samples)
        return Tables.rows(samples)
    end

    return samples
end
