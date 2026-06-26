using ModelingToolkit
using OrdinaryDiffEq
using SymbolicIndexingInterface
using SciMLStructures: Tunable, replace, Initials, Constants, canonicalize

"""
    Model(model_def, sys)

Runtime wrapper around a `ModelDefinition` and compiled ModelingToolkit system.

`ModelDefinition` stores the declarative YAML-derived model. `Model` adds the
mutable runtime state needed by the later stages: the cached `ODEProblem`,
warmup and production parameter setters, uncertain-parameter priors and initial
values, and design-stage setter data.

Create a `Model` after compiling the equations:

```julia
model_def = load_model("model.yml")
@mtkcompile sys = System(model_def.equations, t)
model = Model(model_def, sys)
```

The same `Model` can then be used for simulation, Turing inference, Thompson
sampling scans, and evaluation scans.
"""

# -------------------------------------------------------------------------
# Struct Definitions
# -------------------------------------------------------------------------

mutable struct Model
    model_def::ModelDefinition
    sys:: Any # Compiled ModellingToolkit system
    prob::Union{Nothing, ODEProblem, Function} # ODE supported first

    tunable_priors::Any
    tunable_initial::Tuple{Vararg{Float64}}
    tunable_symbols::Tuple{Vararg{Symbol}}

    # for setting the warmup parameters in the parameter container
    warmup_setter!::Any
    # the ordered warmup values for the setter
    warmup_values::Tuple{Vararg{Float64}}

    # for setting the tunable parameters
    multiparam_setter!::Any
    # the ordered "lists" of values for the setter
    # the first tuple has the values across all parameters for the first part
    multiparam_values::Tuple{Vararg{Tuple}}
    multiparam_constants_only::Bool

    ## separately prepare the setters for design stage
    design_warmup_setter!::Any
    design_warmup_values::Tuple{Vararg{Float64}}
    design_multiparam_setter!::Any
    design_multiparam_values::Tuple{Vararg{Tuple}}
    design_multiparam_constants_only::Bool
    
    # Constructor
    function Model(model_def::ModelDefinition, sys::Any)
        new(model_def, sys, nothing, 
        # tunables
        nothing, () :: Tuple{Vararg{Float64}}, () :: Tuple{Vararg{Float64}},
        # warmup
        nothing, () :: Tuple{Vararg{Float64}},
        # multiparams
        nothing, () :: Tuple{Vararg{Float64}}, false,
        # design warmup
        nothing, () :: Tuple{Vararg{Float64}},
        # design multiparam
        nothing, () :: Tuple{Vararg{Float64}}, false,
        )
    end
end
# ----------------------------------------------------------s---------------
# Helpers
# -------------------------------------------------------------------------

"""
    get_uncertain_parameters(model::Model) -> Vector{Symbol}

Get the parameter names marked as `:uncertain` in the model definition.

These are the parameters that become tunable ModelingToolkit parameters and are
sampled during `TuringSpec` inference.
"""
function get_uncertain_parameters(model::Model)
    uncertain = Symbol[]
    for (name, spec) in model.model_def.parameters
        if spec.role == :uncertain
            push!(uncertain, name)
        end
    end
    return uncertain
end

"""
    get_warmup_params(parameters; design=false)

Return parameter warmup values keyed by symbol.

Warmup values are used for the optional first solve that brings the model to a
starting state before the production solve. With `design=true`, values are read
from each parameter's nested `design` metadata instead of the ordinary YAML
warmup field.
"""
function get_warmup_params(parameters::Dict{Symbol, ParameterSpec}; design::Bool=false)::Dict{Symbol, Float64}
    # check for a warm up stage, and start with warm up values
    warmup_map = Dict{Symbol, Float64}()
    for kv in pairs(parameters)
        warmup_value = if design
            isnothing(kv.second.design) ? nothing : kv.second.design.warmup_value
        else
            kv.second.warmup_value
        end

        if !isnothing(warmup_value)
            warmup_map[kv.first] = warmup_value
        end
    end

    return warmup_map
end

"""
    get_warmup_setter_inputs(model; design=false)

Return ordered warmup values and symbols for `setp`.
"""
function get_warmup_setter_inputs(model::Model; design::Bool=false)
    return get_warmup_setter_inputs(model.sys, model.model_def.parameters; design)
end

"""
    get_warmup_setter_inputs(sys, parameters; design=false)

Return system-ordered warmup setter inputs.
"""
function get_warmup_setter_inputs(sys, parameters::Dict{Symbol, ParameterSpec}; design::Bool=false)
    warmup_map = get_warmup_params(parameters; design)
    ordered_params = ModelingToolkit.parameters(sys)

    warmup_Nums = Vector{Num}(undef, length(warmup_map))
    warmup_values = Vector{Float64}(undef, length(warmup_map))

    counter = 1
    for p in ordered_params
        s = Symbolics.tosymbol(p)

        if haskey(warmup_map, s)
            warmup_Nums[counter] = getproperty(sys, s)
            warmup_values[counter] = warmup_map[s]
            counter += 1
        end
    end

    return (warmup_values=Tuple(warmup_values), warmup_Nums=warmup_Nums)
end

const MultiparamValue = Union{Float64, Tuple{Vararg{Float64}}}

"""
    extract_multiparams(parameters; design=false)

Return scalar or staged parameter values that need production solves.

Tuple-valued parameters define staged production solves. For example, when two
parameters each provide three values, `simulate!` runs three production solves,
pairing values positionally after any warmup solve. In design mode, design
values are included for scanned design parameters while uncertain parameters
remain controlled by posterior samples.
"""
function extract_multiparams(parameters::Dict{Symbol, ParameterSpec}; design::Bool=false)::Dict{Symbol, MultiparamValue}
    multiparams = Dict{Symbol, MultiparamValue}()
    for kv in pairs(parameters)
        has_design_value = design && !isnothing(kv.second.design)
        value = if has_design_value
            kv.second.design.value
        else
            kv.second.value
        end

        if design
            if !isnothing(value) && (has_design_value || kv.second.role != :uncertain)
                multiparams[kv.first] = value
            end
        elseif value isa Union{AbstractArray, Tuple}
            multiparams[kv.first] = Tuple(value)
        end
    end

    # check that all array params have the same length
    lengths = [length(v) for v in values(multiparams) if v isa Union{AbstractArray, Tuple}]
    if length(unique(lengths)) > 1
        error("Parameters have value arrays which differ in length. The lengths are", lengths)
    end

    multiparams
end

"""
    get_multiparam_setter_inputs(model; design=false)

Return ordered multiparameter values and symbols for `setp`.
"""
function get_multiparam_setter_inputs(model::Model; design::Bool=false)
    return get_multiparam_setter_inputs(model.sys, model.model_def.parameters; design)
end

"""
    get_multiparam_setter_inputs(sys, parameters; design=false)

Return system-ordered multiparameter setter inputs.
"""
function get_multiparam_setter_inputs(sys, parameters::Dict{Symbol, ParameterSpec}; design::Bool=false)
    multiparams = extract_multiparams(parameters; design)
    ordered_params = ModelingToolkit.parameters(sys)

    multiparam_Nums = Vector{Num}(undef, length(multiparams))
    multiparam_values = Vector{MultiparamValue}(undef, length(multiparams))

    counter = 1
    for p in ordered_params
        s = Symbolics.tosymbol(p)

        if haskey(multiparams, s)
            multiparam_Nums[counter] = getproperty(sys, s)
            multiparam_values[counter] = multiparams[s]
            counter += 1
        end
    end

    if isempty(multiparams)
        return (multiparam_values=(), multiparam_Nums=multiparam_Nums)
    end

    lengths = [length(v) for v in multiparam_values if v isa Union{AbstractArray, Tuple}]
    multiparam_length = isempty(lengths) ? 1 : only(unique(lengths))

    expanded_values = map(multiparam_values) do value
        value isa Union{AbstractArray, Tuple} ? Tuple(value) : ntuple(_ -> value, multiparam_length)
    end
    multiparam_values_flipped = [collect(x) for x in zip(expanded_values...)]

    return (
        multiparam_values=Tuple(Tuple.(multiparam_values_flipped)),
        multiparam_Nums=multiparam_Nums,
    )
end

"""
    _multiparam_nums_are_constants(sys, multiparam_Nums) -> Bool

Return whether all staged parameters are constants in the system.
"""
function _multiparam_nums_are_constants(sys, multiparam_Nums)::Bool
    isempty(multiparam_Nums) && return false

    for num in multiparam_Nums
        idx = SymbolicIndexingInterface.parameter_index(sys, num)
        if isnothing(idx) || !(idx.portion isa Constants)
            return false
        end
    end

    return true
end

# -------------------------------------------------------------------------
# Inference hook
# -------------------------------------------------------------------------

function evaluate_model(model::Model, p_vec)
    if model.simulation_context === nothing
        error("Model not prepared for simulation. Call setup_simulation!")
    end

    ctx = model.simulation_context

    new_p = model.buffer_func(p_vec)
    
    model.uncertain_param_setter!(new_p, p_vec)

    prob_new = remake(model.prob; p=new_p)

    sol = solve(prob_new, ctx.solver; 
                dt=ctx.dt, 
                saveat=ctx.t_obs, 
                save_idxs=observed_state_index(model.sys, ctx))
    
    return Array(sol)
end

# -------------------------------------------------------------------------
# Simulators
# -------------------------------------------------------------------------



"""
    simulate!(model, simulation; kwargs...)
    simulate!(model, initial_conditions, tspan; kwargs...)

Run the simulation/build stage for a model.

`simulate!` builds and caches the underlying `ODEProblem` on the first call by
delegating to `setup_simulation!`. Later calls reuse that problem and the
precomputed parameter setters, which is why inference and scan stages can call
it repeatedly inside tight loops.

When passed a `SimulationSpec`, `simulate!` uses the spec's initial conditions,
time span, solver, observation times, observed states, solver options, and
uncertain parameter values. Extra keyword arguments override values from the
spec and are forwarded to the lower-level method.

The simulation uses YAML/default parameter values plus any supplied overrides:

- `parameters`: scalar parameter overrides used when the problem is first built.
- `solver` and `solver_opts`: solver object and keyword options for `solve`.
- `saveat` and `save_idxs`: output times and observed state indices.
- `sampled_uncertain_params`: posterior draw values in model tunable order.
- `parameter_setter` and `parameter_values`: low-level setter used by grid scans.
- `design`: use design-stage warmup and multiparameter values.
- `return_simulate`: return `(warmup_sol, sols)` instead of only production
  solutions.

The return value is a vector of production solutions. If staged parameters are
present, the vector contains one solution per stage. With `return_simulate=true`,
the returned named tuple also includes the warmup solution, or `nothing` if no
warmup values were configured.
"""
function simulate!(model::Model, 
                   initial_conditions::Tuple{Vararg{Float64}},
                   tspan::Tuple{Float64, Float64}
                   ;
                   parameters::Dict=Dict{Symbol,Float64}(),
                   solver = Rosenbrock23(),
                   saveat::Vector{Float64} = Float64[],
                   # solve kwargs
                   solver_opts::NamedTuple = (;),
                   save_idxs::Any = nothing,
                   sampled_uncertain_params::Union{Nothing, AbstractVector} = nothing,
                   parameter_setter::Any = nothing,
                   parameter_values::Any = nothing,
                   multiparam_values:: Union{Nothing, Vector{Float64}} = nothing,
                   prealloc_results_vector::Union{Nothing, Vector{SciMLBase.ODESolution}} = nothing,
                   return_simulate::Bool = false,
                   design::Bool = false,
                   )

    # build the problem once
    if isnothing(model.prob)
        @debug "Setting up simulation"
        setup_simulation!(model, 
                        initial_conditions, 
                        tspan;
                        parameters
                        )
    end

    prob = model.prob
    warmup_values = design ? model.design_warmup_values : model.warmup_values
    warmup_setter! = design ? model.design_warmup_setter! : model.warmup_setter!
    stage_multiparam_values = design ? model.design_multiparam_values : model.multiparam_values
    stage_multiparam_setter! = design ? model.design_multiparam_setter! : model.multiparam_setter!
    stage_multiparam_constants_only = design ? model.design_multiparam_constants_only : model.multiparam_constants_only

    multiparam_length = isempty(stage_multiparam_values) ? 1 : length(stage_multiparam_values)
    if isnothing(prealloc_results_vector)
        prealloc_results_vector = Vector{SciMLBase.ODESolution}(undef, multiparam_length)
    end

    if !isnothing(sampled_uncertain_params)
        p_work = replace(Tunable(), prob.p, sampled_uncertain_params)
    else
        ## NON-INFERENCE Sim
        # a deep copy because we are modifying a cuma parameter
        # ? - we could set the cuma parameter here accordingly (in the warmup stage)
        # this way we won't have to rely on the "original container settings"
        p_work = deepcopy(prob.p)
    end

    if !isnothing(parameter_setter)
        parameter_setter(p_work, parameter_values)
    end

    warmup_sol = nothing
    if !isempty(warmup_values)
        # Reset warmup parameters on every simulation call. Some parameter containers
        # returned by `replace(Tunable(), ...)` share non-tunable storage with `prob.p`,
        # so production-stage multiparameter updates can otherwise leak into the next
        # warmup solve.
        warmup_setter!(p_work, warmup_values)

        warmup_sol = solve(prob, solver, p=p_work; solver_opts..., save_end=true, save_everystep=false, dense=false)
        p_work = replace(Initials(), p_work, warmup_sol.u[end])

        if !isnothing(parameter_setter)
            parameter_setter(p_work, parameter_values)
        end
    end

    opts_prod = solver_opts
    opts_prod = isempty(saveat) ? opts_prod : merge(opts_prod, (saveat=saveat, ))
    opts_prod = isnothing(save_idxs) ? opts_prod : merge(opts_prod, (save_idxs=save_idxs, ))

    use_threaded_constants = (
        multiparam_length > 1 &&
        Base.Threads.nthreads() > 1 &&
        stage_multiparam_constants_only &&
        isnothing(parameter_setter)
    )

    if use_threaded_constants
        # Each production solve only changes constants such as `cuma`; isolate that
        # storage per thread and remake the problem so ODE internals are not shared.
        base_constants = canonicalize(Constants(), p_work)[1]

        Base.Threads.@threads for i in 1:multiparam_length
            p_i = replace(Constants(), p_work, collect(base_constants))
            stage_multiparam_setter!(p_i, stage_multiparam_values[i])
            prob_i = remake(prob; p=p_i)
            sol = solve(prob_i, solver; opts_prod...)
            prealloc_results_vector[i] = sol
        end
    else
        for i in 1:multiparam_length
            # set all multiparameters
            if !isempty(stage_multiparam_values)
                stage_multiparam_setter!(p_work, stage_multiparam_values[i])
            end

            if !isnothing(parameter_setter)
                parameter_setter(p_work, parameter_values)
            end

            sol = solve(prob, solver; p=p_work, opts_prod...)

            prealloc_results_vector[i] = sol
        end
    end

    if return_simulate
        return (warmup_sol=warmup_sol, sols=prealloc_results_vector)
    end

    return prealloc_results_vector
end


"""
    setup_simulation!(model, initial_conditions, tspan; parameters=Dict())

Build and cache the ODE problem plus parameter setters used by `simulate!`.

This function is the explicit build step. It creates the `ODEProblem`, collects
scalar parameter values from YAML and `parameters`, prepares warmup setters,
detects staged production parameters, builds design-stage setters, and records
the tunable uncertain parameters in the compiled system order expected by
Turing.

Most users can call `simulate!`, `run_inference`, or `run_scan` directly and let
them call this function as needed. Call it yourself when you want to pay the
build cost up front or inspect the prepared model fields before running a later
stage.
"""
function setup_simulation!(model::Model,
                          initial_conditions::Tuple{Vararg{Float64}},
                          tspan::Tuple{Float64, Float64};
                          parameters::Dict=Dict{Symbol,Float64}(),
                          )
    
    # Get states from the compiled system
    # FIXME - Initial conditions should be named tuples (do not rely on indices)
    u0 = Dict(unknowns(model.sys) .=> initial_conditions)

    # Get all the parameters and their values as pairs for input into the problem - like mtk expects
    # BUT ovveride with new starting values if they have been provided
    p_map = Dict{Symbol, Float64}()
    uncertain_param_symbols = Vector{Symbol}()

    # FIXME
    # ideally we'd have a function "get initial values", that's the first value or a warmup

    warmup_map = get_warmup_params(model.model_def.parameters)
    if !isempty(warmup_map)
        @info "Warm up parameters present. Setting initial warmup values. "
    end

    # For all uncertain parameters in the model_definition
    for(name, param_spec) in model.model_def.parameters
        if param_spec.role == :uncertain
            push!(uncertain_param_symbols, name)
        end
        
        val = param_spec.value

        # if it is an array we skip this parameter as it will be set later
        if val isa AbstractArray || val isa Tuple
            continue
        end
        
        if val !== nothing
            p_map[name] = val
        end
    end

    # This creates the dictionary that MTK needs to build the problem
    params = merge(p_map, warmup_map)

    p_map_vars = Dict(
        getproperty(model.sys, name) => val for (name, val) in params
    )

    # Create the problem with all parameters and their starting values
    model.prob = ODEProblem(model.sys, merge(u0,p_map_vars), tspan, jac=true)

    ## warmup
    warmup_values, warmup_Nums = get_warmup_setter_inputs(model)
    model.warmup_values = warmup_values
    model.warmup_setter! = setp(model.sys, warmup_Nums)

    design_warmup_values, design_warmup_Nums = get_warmup_setter_inputs(model; design=true)
    model.design_warmup_values = design_warmup_values
    model.design_warmup_setter! = setp(model.sys, design_warmup_Nums)

    ## multiparams
    multiparam_values, multiparam_Nums = get_multiparam_setter_inputs(model)
    model.multiparam_values = multiparam_values
    model.multiparam_setter! = isempty(multiparam_Nums) ? nothing : setp(model.sys, multiparam_Nums)
    model.multiparam_constants_only = _multiparam_nums_are_constants(model.sys, multiparam_Nums)

    design_multiparam_values, design_multiparam_Nums = get_multiparam_setter_inputs(model; design=true)
    model.design_multiparam_values = design_multiparam_values
    model.design_multiparam_setter! = isempty(design_multiparam_Nums) ? nothing : setp(model.sys, design_multiparam_Nums)
    model.design_multiparam_constants_only = _multiparam_nums_are_constants(model.sys, design_multiparam_Nums)

    ## tunable
    ordered_params = ModelingToolkit.parameters(model.sys)
    tunable_params = [p for p in ordered_params if p in Set(ModelingToolkit.tunable_parameters(model.sys))]

    model.tunable_symbols = Tuple(Symbolics.tosymbol(p) for p in tunable_params)
    tunable_priors = make_priors(model)
    model.tunable_priors = arraydist(tunable_priors)
    model.tunable_initial = get_initial_tunables(model)
    
    @info "Model built and compiled..."

    return nothing
end

"""
    make_priors(model) -> Vector

Build priors for tunable uncertain parameters in system order.

The order is the compiled ModelingToolkit tunable-parameter order, not
necessarily the order in the YAML file. `run_inference` uses the same order for
posterior draws and then renames sampled chain columns back to parameter
symbols.
"""
function make_priors(model::Model)::Vector{Uniform{Float64}}
    ordered_params = ModelingToolkit.parameters(model.sys)
    tunable_params = [p for p in ordered_params if p in Set(ModelingToolkit.tunable_parameters(model.sys))]
    
    priors = Vector{Uniform{Float64}}(undef, length(tunable_params))

    for (i, param) in enumerate(tunable_params)
        s = Symbolics.tosymbol(param)

        for (param_symbol, param_spec) in model.model_def.parameters
            if param_symbol == s
                if param_spec.role != :uncertain
                    error("A found uncertain parameter $s is not uncertain")
                end

                priors[i] = make_prior(param_spec.prior)
            end
        end
    end

    return priors
end

"""
    make_prior(prior) -> Distribution

Build a prior distribution from YAML metadata.

Currently supported metadata:

```yaml
prior:
  distribution: uniform
  lower: 0.0
  upper: 1.0
```
"""
function make_prior(prior::Dict)
    dist = lowercase(prior["distribution"])
    if dist == "uniform"
        return Distributions.Uniform(prior["lower"], prior["upper"])
    end
    
    error("Unsupported prior distribution: $(prior["distribution"])")
end

"""
    get_initial_tunables(model) -> Tuple

Return initial uncertain parameter values in system tunable order.

These values seed Turing through `make_initial_params`. Each uncertain
parameter must have a scalar YAML `value`; tuple-valued uncertain parameters are
not supported for Turing initialisation.
"""
function get_initial_tunables(model::Model)::Tuple{Vararg{Float64}}
    ordered_params = ModelingToolkit.parameters(model.sys)
    tunable_params = [p for p in ordered_params if p in Set(ModelingToolkit.tunable_parameters(model.sys))]

    initial_tunable_values = Vector{Float64}(undef, length(tunable_params))

    for (i, param) in enumerate(tunable_params)
        s = Symbolics.tosymbol(param)

        for (param_symbol, param_spec) in model.model_def.parameters
            if param_symbol == s
                if isnothing(param_spec.value)
                    error("No initial value found for uncertain parameter $s. Provide it in the YAML or via spec.simulation.uncertain_param_values.")
                elseif param_spec.value isa Tuple || param_spec.value isa AbstractArray
                    error("Uncertain parameter $s has a non-scalar initial value, which is not supported for Turing initialisation.")
                end

                initial_tunable_values[i] = float(param_spec.value)
            end
        end
    end

    return Tuple(initial_tunable_values)
end

"""
    validate_initial_tunables(symbols, initial_values, priors)

Check that initial tunable values are inside their prior support.
"""
function validate_initial_tunables(symbols, initial_values, priors)
    for (s, initial_value, prior) in zip(symbols, initial_values, priors)
        if !isfinite(logpdf(prior, initial_value))
            error("Initial value for uncertain parameter $s is outside its prior support: value=$initial_value, prior=$prior.")
        end
    end

    return nothing
end
