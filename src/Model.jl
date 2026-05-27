using ModelingToolkit
using OrdinaryDiffEq
using SymbolicIndexingInterface
using SciMLStructures: Tunable, canonicalize, replace, replace!, Initials
using PreallocationTools
using Plots
using ForwardDiff

"""
Model

A simulation wrapper for the ModelDefinition that creates and manages ODE Problems
- can extend to other problems later

Provides methods for simulation, parameter manipulation, and result visualisation

"""

# -------------------------------------------------------------------------
# Struct Definitions
# -------------------------------------------------------------------------

mutable struct Model
    model_def::ModelDefinition
    sys:: Any # Compiled ModellingToolkit system
    prob::Union{Nothing, ODEProblem, Function} # ODE supported first

    tunable_priors::Any

    # for setting the warmup parameters in the parameter container
    warmup_setter!::Any
    # the ordered warmup values for the setter
    warmup_values::Union{Nothing, Tuple{Vararg{Symbol}}}

    # for setting the tunable parameters
    multiparam_setter!::Any
    # the ordered "lists" of values for the setter
    multiparam_values::Union{Nothing, Tuple{Vararg{Tuple}}}
    
    # Constructor
    function Model(model_def::ModelDefinition, sys::Any)
        new(model_def, sys, nothing, 
        nothing, nothing,
        nothing, nothing)
    end
end
# ----------------------------------------------------------s---------------
# Helpers
# -------------------------------------------------------------------------

"""
    get_uncertain_parameters(model::Model) -> Vector{Symbol}

Get the parameter names marked as :uncertain in the model definition

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

function get_warmup_params(parameters::Dict{Symbol, ParameterSpec}):: Dict{Symbol, Float64}
    # check for a warm up stage, and start with warm up values
    warmup_map = Dict{Symbol, Float64}()
    for kv in pairs(parameters)
        if !isnothing(kv.second.warmup_value)
            warmup_map[kv.first] = kv.second.warmup_value
        end
    end

    return warmup_map
end

function get_array_params(parameters::Dict{Symbol, ParameterSpec})::Dict{Symbol, Tuple{Vararg{Float64}}}
    multiparams = Dict{Symbol, Tuple{Vararg{Float64}}}()
    for kv in pairs(parameters)
        if kv.second.value isa Union{AbstractArray, Tuple}
            multiparams[kv.first] = kv.second.value
        end
    end

    # check that all array params have the same length
    lengths = [length(v) for v in values(multiparams)]
    if length(unique(lengths)) > 1
        error("Parameters have value arrays which differ in length. The lengths are", lengths)
    end

    multiparams
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
                save_idxs=ctx.obs_state_idx)
    
    return Array(sol)
end

# -------------------------------------------------------------------------
# Simulators
# -------------------------------------------------------------------------



"""
Simulate

    simulate!(model::Model, 
              initial_conditions::Vector{Float64},
              parameters::Dict,
              tspan::Tuple{Float64, Float64};
              solver=Tsit5(),
              dt::Float64=0.01,
              saveat=Float64[])

Runs a simple one off simulation and stores the results

# Arguments
- `model`: The Model object to simulate
- `initial_conditions`: Vector of initial values for each state variable
- `parameters`: Dict mapping parameter symbols to values
- `tspan`: Time span as (t_start, t_end)

# Keyword Arguments
- `solver`: ODE solver algorithm (default: Tsit5())
- `dt`: Time step for solver (default: 0.01)
- `saveat`: Specific time points to save (default: all points)

# Returns
- The solution object (also stored in model.sol)

SHOULD NOT MAKE ALTER THE MODEL!

"""


function simulate!(model::Model, 
                   initial_conditions::Tuple{Vararg{Float64}},
                   tspan::Tuple{Float64, Float64}
                   ;
                   parameters::Dict=Dict{Symbol,Float64}(),
                   solver = Rosenbrock23(),
                   saveat::Any = Float64[], # TODO - change to vector 
                   # solve kwargs
                   solver_opts::NamedTuple = (;),
                   save_idxs::Any = nothing,
                   sampled_uncertain_params::Union{Nothing, AbstractVector} = nothing,
                   multiparam_values:: Union{Nothing, Vector{Float64}} = nothing,
                   multiparam_length:: Int = 1,
                   prealloc_results_vector::Union{Nothing, Vector{SciMLBase.ODESolution}} = nothing
                   )

    # build the problem once
    if isnothing(model.prob)
        println("set up simulation")
        setup_simulation!(model, 
                        initial_conditions, 
                        parameters, 
                        tspan, 
                        solver=solver, 
                        solver_opts=solver_opts)
    end

    prob = model.prob

    if !isnothing(sampled_uncertain_params)
        ## INFERENCE
        # the tunable parameters have to be in the right order
        p_work = replace(Tunable(), prob.p, sampled_uncertain_params)
    else
        ## NON-INFERENCE Sim
        # FIXME - we take a deep copy because we are modifying a cuma parameter
        # ?we could set the cuma parameter here accordingly (in the warmup stage)
        # this way we won't have to rely on the "original container settings"
        p_work = deepcopy(prob.p)

        # refactor this with your other function
        multiparams = model.multiparams
        @show multiparams
        multiparam_count = isempty(multiparams) ? 1 : length(keys(multiparams))
        multiparam_values = Vector{Float64}(undef, multiparam_count)
        for (i, symbol) in enumerate(model.multiparam_symbols)
            # TODO check if warm up has these parameters
            # TODO this is no longer necessary as we have a specific setter up now for multiparam
            if symbol in keys(model.warmup_params)
                multiparam_values[i] = model.warmup_params[symbol]
            else
                multiparam_values[i] = multiparams[symbol][1]
            end
        end

        # FIXME - consider an inner function depending on the parameters present, but most likely this preallocaiton is redundant
        prealloc_results_vector = Vector{SciMLBase.ODESolution}(undef, multiparam_length)
    end

    if !isnothing(model.warmup_settable)
        warm = solve(prob, solver, p=p_work; solver_opts..., save_end=true, save_everystep=false, dense=false)
        p_work = replace(Initials(), p_work, warm.u[end])        
        # TODO - add the check if the values you modify are indeed indexes 1 and 2 
        #        some Julia versions move the actual u0 to indexes 2, 3 which breaks replace(Initials()...)
    end

    opts_prod = solver_opts
    opts_prod = isempty(saveat) ? opts_prod : merge(opts_prod, (saveat=saveat, ))
    opts_prod = isnothing(save_idxs) ? opts_prod : merge(opts_prod, (save_idxs=save_idxs, ))

    for i in 1:multiparam_length

        # set all multiparameters
        for (j, symbol) in enumerate(model.multiparam_symbols)
            multiparam_values[j] = model.multiparams[symbol][i]
        end

        @show p_work    
        @show multiparam_values
        model.multiparam_setter!(p_work, multiparam_values)

        sol = solve(prob, solver; p=p_work, opts_prod...)

        prealloc_results_vector[i] = sol
    end

    return prealloc_results_vector
end


"""
Prepares the model for simulation, created onced for many evaluations.

    setup_evaluation!(model::Model;
                      t_obs::Vector{Float64},
                      obs_state_idx::Int,
                      initial_conditions::Vector{Float64},
                      tspan::Tuple{Float64, Float64},
                      solver=Euler(),
                      dt::Float64=0.01)

"""

function setup_simulation!(model::Model,
                          initial_conditions::Tuple{Vararg{Float64}},
                          tspan::Tuple{Float64, Float64};
                          solver::Any=Euler(),
                          solver_opts::Union{Nothing, NamedTuple}=NamedTuple(),
                          t_obs::Union{Vector{Float64}, Nothing}=nothing,
                          obs_state_idx::Union{Int, Nothing}=nothing,
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

    # prepare the symbols for multiparams
    multiparams_Nums = Vector{Num}(undef, length(model.multiparams))
    # TODO note that order matters
    multiparam_symbols = Tuple(collect(keys(model.multiparams)))
    for (i, symbol) in enumerate(multiparam_symbols)
        # FIXME - this should be iterated using the internal order of the MTK system
        # not the arbitrary multiparam_symbols
        multiparams_Nums[i] = getproperty(model.sys, symbol)
    end

    # prepare cuma setter and priors
    tunable_params = [p for p in ordered_params if p in Set(ModelingToolkit.tunable_parameters(ns))]
    cuma_setter! = setp(ns, [getproperty(ns, :cuma),])


    # TODO mark as settable ordered
    model.multiparam_symbols = multiparam_symbols
    
    # TODO - finding warmup ideally is done after being set in model struct
    warmup_settable = Vector{Pair{Int32, Float64}}(undef, length(warmup_map))
    for (i, (warmup_symbol, warmup_value)) in enumerate(warmup_map)
        warmup_index = findfirst(==(warmup_symbol), settable_symbols)
        warmup_settable[i] = warmup_index => warmup_value
        if !(warmup_symbol in settable_symbols)
            # TODO account for this case
            error("the warmup parameters is not in settable params")
        end
    end
    model.warmup_settable = warmup_settable

    model.multiparam_setter! = setp(model.sys, multiparams_Nums)

    model.settable_symbols = settable_symbols
    
    # fixme - this is a spec now
    model.simulation_context = (
        t_obs = t_obs,
        obs_state_idx = obs_state_idx,
        solver = solver,
        solver_opts = solver_opts)

    @info "Model built and compiled..."


    return nothing
end