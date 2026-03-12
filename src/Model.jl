using ModelingToolkit
using OrdinaryDiffEq
using SymbolicIndexingInterface
using SciMLStructures: Tunable, canonicalize, replace, replace!
using PreallocationTools
using Plots
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
    prob::Union{Nothing, ODEProblem, Function} #ODE supported first
    sol::Union{Nothing, Any} # solution of the LAST simulation

    warmup::Bool

    # fields for inference procedure
    param_setter:: Union{Nothing, Any}
    buffer_func::Union{Nothing, Function}
    uncertain_params::Union{Nothing, Vector}
    simulation_context::Union{Nothing, NamedTuple}

    # Constructor
    function Model(model_def::ModelDefinition, sys::Any)
        # does warmup exist
        warmup_params = get_warmup_params(model_def.parameters)
        warmup = !isempty(warmup_params)

        #Problem and solution are initially empty as they are created during simulation
        new(model_def, sys, nothing, nothing, warmup, nothing, nothing, nothing, nothing)

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

function get_warmup_params(parameters::Dict{Symbol, ParameterSpec})
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
    
    model.param_setter(new_p, p_vec)

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
                   tspan::Tuple{Float64, Float64};

                   parameters::Dict=Dict{Symbol,Float64}(),
                   solver=Rosenbrock23(),
                #    dt::Float64=nothing,
                   saveat=Float64[],
                   # solve kwargs
                   solver_opts::NamedTuple = NamedTuple(),
                   save_idxs=nothing,
                   )

    # build the problem once
    if isnothing(model.prob)
        setup_simulation!(model, 
                        initial_conditions, 
                        parameters, 
                        tspan, 
                        solver=solver, 
                        solver_opts=solver_opts)
    end

    prob = model.prob

    # update the parameters as requested
    if !isempty(parameters)
        prob = remake(prob; p=parameters)
    end

    u0 = initial_conditions
    if model.warmup
        # initial params are the warm up params
        sol = solve(prob, solver; solver_opts...)

        # overwrite u0 for production run
        u0 = sol.u[end]
    end

    # prepare parameters
    multiparams = get_array_params(model.model_def.parameters)
    param_len = isempty(multiparams) ? 1 : length(first(values(multiparams)))

    opts_prod = solver_opts
    opts_prod = isempty(saveat) ? opts_prod : merge(opts_prod, (saveat=saveat, ))
    opts_prod = isnothing(save_idxs) ? opts_prod : merge(opts_prod, (save_idxs=save_idxs, ))

    results = Vector{SciMLBase.ODESolution}()
    for i in 1:param_len
        # prepare the parameters for the next run
        p_symbol_dict = Dict()
        for (k, v) in multiparams
            p_symbol_dict[k] = v[i]
        end

        prob_i = remake(model.prob, u0=u0, p=p_symbol_dict)

        sol = solve(prob_i, solver; opts_prod...)
        
        push!(results, sol)
    end

    return results
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
                          uncertain_param_values::Any,
                          tspan::Tuple{Float64, Float64};
                          solver::Any=Euler(),
                          solver_opts::NamedTuple=NamedTuple(),
                          t_obs::Union{Vector{Float64}, Nothing}=nothing,
                          obs_state_idx::Union{Int, Nothing}=nothing,
                          )
    
    # Get states from the compiled system
    u0 = Dict(unknowns(model.sys) .=> initial_conditions)
    # Get all the parameters and their values as pairs for input into the problem - like mtk expects
    # BUT ovveride with new starting values if they have been provided
    p_map = Dict{Symbol, Float64}()
    uncertain_param_names = []

    warmup_map = get_warmup_params(model.model_def.parameters)
    if !isempty(warmup_map)
        @info "Warm up parameters present. Using initial warmup values. "
    end

    # For all uncertain parameters in the model_definition
    for(name, param_spec) in model.model_def.parameters
        if param_spec.role == :uncertain
            push!(uncertain_param_names, name)
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

    # Now check if the user has provided new values
    for (param_name, param_value) in uncertain_param_values
        p_map[param_name] = param_value
    end

    # This creates the dictionary that MTK needs to build the problem
    params = merge(p_map, warmup_map)

    p_map_vars = Dict(
        getproperty(model.sys, name) => val
        for (name, val) in params
    )

    # Create the problem with all parameters and their starting values - including user provided ones
    model.prob = ODEProblem(model.sys, merge(u0, p_map_vars), tspan)
    
    uncertain_syms = Vector{Any}(undef, length(uncertain_param_names))

    for (i, name) in enumerate(uncertain_param_names)
        uncertain_syms[i] = getproperty(model.sys, name)
    end

    model.uncertain_params =  uncertain_syms

    model.param_setter = setp(model.sys, uncertain_syms)

    model.buffer_func = (p) -> remake_buffer(
        model.sys, model.prob.p, Dict(zip(uncertain_syms, p))
    )
    
    model.simulation_context = (
        t_obs = t_obs,
        obs_state_idx = obs_state_idx,
        solver = solver,
        solver_opts = solver_opts)

    @info "Model built and compiled..."


    return nothing
end