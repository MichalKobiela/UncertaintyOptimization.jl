using Tables

"""
    run_scan(samples, spec::GridScan, model::Model; evaluator = nothing)

Run a one-dimensional grid scan for each sample in `samples` using `spec.values`.

By default, each candidate is evaluated with `simulate!` using the settings from
`spec.simulation`. The posterior sample is converted into the model's tunable parameter
order, `spec.scale` is updated according to `spec.kind`, and the loss is evaluated as:

```julia
sim = simulate!(model, ...; sampled_uncertain_params=params, design=true, return_simulate=true)
spec.lossf(sim.warmup_sol, predicted_sol; sys=model.sys)
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
    if isempty(spec.values)
        error("GridScan.values must not be empty")
    end

    if isnothing(model.prob)
        setup_model_for_simulation(model, spec.simulation)
    end

    posterior_samples = _scan_rows(samples)
    results = NamedTuple[]

    for (sample_index, one_posterior) in enumerate(posterior_samples)
        losses = Vector{Float64}(undef, length(spec.values))

        for (grid_index, grid_value) in pairs(spec.values)
            # use the simulate function to evaluate 
            # first, you have to scale the value
            # simulate!()

            evaluated = if isnothing(evaluator)
                _default_grid_scan_evaluator(one_posterior, grid_value, spec, model)
            else
                evaluator(one_posterior, grid_value, spec)
            end

            loss_value = _call_loss(spec.lossf, evaluated, model.sys)

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
            kind = spec.kind,
            best_value = spec.values[best_index],
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

function _call_loss(lossf, evaluated, sys)
    args = evaluated isa Tuple ? evaluated : (evaluated,)

    try
        return lossf(args...; sys=sys)
    catch err
        if _unsupported_sys_keyword(err, lossf)
            return lossf(args...)
        end

        rethrow()
    end
end

function _unsupported_sys_keyword(err, lossf)
    err isa MethodError || return false
    err.f === Core.kwcall || return false
    length(err.args) >= 2 || return false

    kws = err.args[1]
    return err.args[2] === lossf && kws isa NamedTuple && haskey(kws, :sys)
end

function _default_grid_scan_evaluator(one_posterior, grid_value, spec::GridScan, model::Model)
    simulation = spec.simulation

    sampled_uncertain_params = _sampled_uncertain_params(model, one_posterior)
    _apply_grid_value!(sampled_uncertain_params, model, spec.scale, grid_value, spec.kind)

    sim = simulate!(
        model,
        simulation.initial_conditions,
        simulation.tspan;
        solver=simulation.solver,
        saveat=simulation.t_obs,
        save_idxs=simulation.obs_state_idx,
        solver_opts=simulation.solver_opts,
        sampled_uncertain_params=sampled_uncertain_params,
        return_simulate=true,
        design=true,
    )

    predicted_sol = length(sim.sols) == 1 ? only(sim.sols) : sim.sols
    return (sim.warmup_sol, predicted_sol)
end

function _sampled_uncertain_params(model::Model, one_posterior)
    if one_posterior isa AbstractVector{<:Real} && length(one_posterior) == length(model.tunable_symbols)
        return Float64.(one_posterior)
    end

    return [
        Float64(_sample_value(one_posterior, symbol))
        for symbol in model.tunable_symbols
    ]
end

function _sample_value(one_posterior, symbol::Symbol)
    if one_posterior isa AbstractDict
        return one_posterior[symbol]
    end

    if hasproperty(one_posterior, symbol)
        return getproperty(one_posterior, symbol)
    end

    try
        return Tables.getcolumn(one_posterior, symbol)
    catch
        try
            return one_posterior[symbol]
        catch
            error("Could not read posterior sample parameter $symbol from $(typeof(one_posterior)).")
        end
    end
end

function _apply_grid_value!(sampled_uncertain_params, model::Model, scale::Union{Nothing, Symbol}, grid_value, kind::Symbol)
    isnothing(scale) && return sampled_uncertain_params

    scale_index = findfirst(==(scale), model.tunable_symbols)
    if isnothing(scale_index)
        error("GridScan.scale $scale is not one of the model tunable parameters: $(model.tunable_symbols).")
    end

    if kind === :scale
        sampled_uncertain_params[scale_index] *= Float64(grid_value)
    elseif kind === :value
        sampled_uncertain_params[scale_index] = Float64(grid_value)
    else
        error("GridScan.kind must be either :scale or :value. Got :$kind.")
    end

    return sampled_uncertain_params
end
