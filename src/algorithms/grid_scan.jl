using Tables
using DataFrames

"""
    run_scan(samples, spec::CartesianScanner, model::Model; evaluator = nothing)

Run a grid scan for each sample in `samples` using the Cartesian product of `spec.scan`.

By default, each candidate is evaluated using the settings from
`spec.simulation`. The posterior sample is converted into the model's tunable parameter
order, each scan axis is updated according to its `kind`, and the loss is evaluated as:

```julia
spec.loss(sim.warmup_sol, predicted_sol; sys=model.sys)
```

If `evaluator` is provided, it is called first as:

```julia
evaluated = evaluator(sample, grid_values, spec)
```

Then the loss is evaluated as:

- `spec.loss(evaluated...)` if `evaluated` is a tuple
- `spec.loss(evaluated)` otherwise

This lets a caller provide an evaluator that turns `(posterior_sample, grid_values, spec)`
into simulation outputs such as `(warmup_sol, predicted_sol)`.

Returns a `DataFrame` with one row per posterior iteration and scan candidate.
Scan values are expanded into columns named from each scan axis, for example
`kx2_scaler` for `(symbol = :kx2, kind = :scale)` and `kx2_value` for
the resolved parameter value used by the simulation. For `(symbol = :kx2,
kind = :value)`, only `kx2_value` is emitted because the grid value is already
the resolved parameter value.
"""
function run_scan(samples, spec::ThompsonGridSpec, model::Model; evaluator = nothing)
    if isnothing(model.prob)
        setup_model_for_simulation(model, spec.simulation)
    end

    scan_plan = _scan_plan(model, spec)
    scan_column_names = _scan_column_names(spec)
    posterior_samples = _scan_rows(samples)
    rows = NamedTuple[]

    for (iteration, one_posterior) in enumerate(posterior_samples)
        sampled_uncertain_params = if isnothing(evaluator) || _scan_output_needs_sampled_uncertain_params(spec, scan_plan)
            _sampled_uncertain_params(model, one_posterior)
        else
            nothing
        end
        losses = Vector{Float64}(undef, length(spec.combinations))

        for (candidate_index, grid_values) in pairs(spec.combinations)

            evaluated = if isnothing(evaluator)
                _default_grid_scan_evaluator(grid_values, spec, model, scan_plan, sampled_uncertain_params)
            else
                evaluator(one_posterior, grid_values, spec)
            end

            loss_value = _call_loss(spec.loss, evaluated, model.sys)

            if !(loss_value isa Real)
                error("CartesianScanner.loss must return a real scalar. Got $(typeof(loss_value)).")
            end

            losses[candidate_index] = Float64(loss_value)
        end

        best_index = argmin(losses)
        best_loss = losses[best_index]

        for (candidate_index, grid_values) in pairs(spec.combinations)
            push!(rows, (
                iteration = iteration,
                candidate_index = candidate_index,
                _scan_values(scan_column_names, spec, grid_values, scan_plan, sampled_uncertain_params)...,
                loss = losses[candidate_index],
                is_best = candidate_index == best_index,
                best_loss = best_loss,
            ))
        end
    end

    return DataFrame(rows)
end

function _scan_column_names(spec::ThompsonGridSpec)
    column_names = Symbol[]
    counts = Dict{Symbol, Int}()

    for axis in spec.scan
        push!(column_names, _unique_scan_column_name!(counts, _scan_grid_column_name(axis)))

        if axis.kind === :scale
            push!(column_names, _unique_scan_column_name!(counts, _scan_value_column_name(axis)))
        end
    end

    return column_names
end

function _unique_scan_column_name!(counts, base_name::Symbol)
    count = get(counts, base_name, 0) + 1
    counts[base_name] = count

    return count == 1 ? base_name : Symbol(base_name, "_", count)
end

function _scan_grid_column_name(axis)
    suffix = axis.kind === :scale ? "scaler" : "value"
    return Symbol(axis.symbol, "_", suffix)
end

function _scan_value_column_name(axis)
    return Symbol(axis.symbol, "_value")
end

function _scan_values(column_names, spec::ThompsonGridSpec, grid_values, scan_plan, sampled_uncertain_params)
    values = Float64[]

    for i in eachindex(spec.scan)
        axis = spec.scan[i]
        push!(values, grid_values[i])

        if axis.kind === :scale
            base_value = _scan_axis_base_value(sampled_uncertain_params, scan_plan, i)
            push!(values, _grid_parameter_value(base_value, grid_values[i], axis.kind))
        end
    end

    return NamedTuple{Tuple(column_names)}(Tuple(values))
end

function _scan_output_needs_sampled_uncertain_params(spec::ThompsonGridSpec, scan_plan)
    return any(eachindex(spec.scan)) do i
        spec.scan[i].kind === :scale && !isnothing(scan_plan.tunable_indices[i])
    end
end

function _scan_axis_base_value(sampled_uncertain_params, scan_plan, axis_index)
    tunable_index = scan_plan.tunable_indices[axis_index]
    return isnothing(tunable_index) ? scan_plan.fixed_values[axis_index] : sampled_uncertain_params[tunable_index]
end

function _scan_rows(samples)
    if Tables.istable(samples)
        return Tables.rows(samples)
    end

    return samples
end

function _call_loss(loss, evaluated, sys)
    args = evaluated isa Tuple ? evaluated : (evaluated,)

    try
        return loss(args...; sys=sys)
    catch err
        if _unsupported_sys_keyword(err, loss)
            return loss(args...)
        end

        rethrow()
    end
end

function _unsupported_sys_keyword(err, loss)
    err isa MethodError || return false
    err.f === Core.kwcall || return false
    length(err.args) >= 2 || return false

    kws = err.args[1]
    return err.args[2] === loss && kws isa NamedTuple && haskey(kws, :sys)
end

function _default_grid_scan_evaluator(grid_values, spec::ThompsonGridSpec, model::Model, scan_plan, sampled_uncertain_params)
    simulation = spec.simulation

    sampled_uncertain_params = copy(sampled_uncertain_params)
    parameter_values = _scan_parameter_values(sampled_uncertain_params, spec, grid_values, scan_plan)

    sim = simulate!(
        model,
        simulation.initial_conditions,
        simulation.tspan;
        solver=simulation.solver,
        saveat=simulation.t_obs,
        save_idxs=observed_state_save_idxs(model.sys, simulation),
        solver_opts=simulation.solver_opts,
        sampled_uncertain_params=sampled_uncertain_params,
        parameter_setter=scan_plan.setter!,
        parameter_values=parameter_values,
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

function _scan_plan(model::Model, spec::ThompsonGridSpec)
    nums = [getproperty(model.sys, axis.symbol) for axis in spec.scan]
    tunable_indices = Vector{Union{Nothing, Int}}(undef, length(spec.scan))
    fixed_values = Vector{Float64}(undef, length(spec.scan))

    for (i, axis) in pairs(spec.scan)
        tunable_index = findfirst(==(axis.symbol), model.tunable_symbols)
        tunable_indices[i] = tunable_index

        if isnothing(tunable_index)
            fixed_values[i] = _scan_base_parameter_value(model, axis.symbol)
        else
            fixed_values[i] = NaN
        end
    end

    return (setter! = setp(model.sys, nums), tunable_indices = tunable_indices, fixed_values = fixed_values)
end

function _scan_base_parameter_value(model::Model, symbol::Symbol)
    if !haskey(model.model_def.parameters, symbol)
        error("CartesianScanner.scan symbol $symbol is not one of the model parameters.")
    end

    value = model.model_def.parameters[symbol].value
    if value isa AbstractArray || value isa Tuple || isnothing(value)
        error("CartesianScanner.scan symbol $symbol must have a scalar base value when it is not tunable.")
    end

    return Float64(value)
end

function _scan_parameter_values(sampled_uncertain_params, spec::ThompsonGridSpec, grid_values, scan_plan)
    parameter_values = Vector{Float64}(undef, length(spec.scan))

    @inbounds for i in eachindex(spec.scan)
        axis = spec.scan[i]
        tunable_index = scan_plan.tunable_indices[i]
        base_value = _scan_axis_base_value(sampled_uncertain_params, scan_plan, i)
        parameter_values[i] = _grid_parameter_value(base_value, grid_values[i], axis.kind)
    end

    return parameter_values
end

function _apply_grid_value!(sampled_uncertain_params, model::Model, symbol::Symbol, grid_value, kind::Symbol)
    symbol_index = findfirst(==(symbol), model.tunable_symbols)
    if isnothing(symbol_index)
        error("CartesianScanner.scan symbol $symbol is not one of the model tunable parameters: $(model.tunable_symbols).")
    end

    return _apply_grid_value_at_index!(sampled_uncertain_params, symbol_index, grid_value, kind)
end

function _apply_grid_value_at_index!(sampled_uncertain_params, symbol_index::Int, grid_value, kind::Symbol)
    sampled_uncertain_params[symbol_index] = _grid_parameter_value(sampled_uncertain_params[symbol_index], grid_value, kind)
    return sampled_uncertain_params
end

function _grid_parameter_value(base_value, grid_value, kind::Symbol)
    if kind === :scale
        return Float64(base_value) * Float64(grid_value)
    elseif kind === :value
        return Float64(grid_value)
    else
        error("CartesianScanner.kind must be either :scale or :value. Got :$kind.")
    end
end
