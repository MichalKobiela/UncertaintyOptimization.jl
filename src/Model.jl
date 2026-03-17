using ModelingToolkit
using OrdinaryDiffEq
using SymbolicIndexingInterface
using SciMLStructures: Tunable, canonicalize, replace, replace!
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
mutable struct TunableP{T}
    tunable_parameters::T
end

mutable struct Model
    # TODO: extarct the MTK specific building blocks into their own sub-struct. 

    model_def::ModelDefinition
    sys:: Any # Compiled ModellingToolkit system
    prob::Union{Nothing, ODEProblem, Function} #ODE supported first
    sol::Any # solution of the LAST simulation

    warmup_params::Union{Nothing, Dict{Symbol, Float64}}
    warmup_settable:: Union{Nothing, Vector{Pair{Int32, Float64}}}
    # TODO - add "ordered" ie order matters
    multiparams::Union{Nothing, Dict{Symbol, Tuple{Vararg{Float64}}}}
    multiparam_symbols::Union{Nothing, Tuple{Vararg{Symbol}}}

    # fields for inference procedure
    param_setter!:: Any
    # TODO - write that this is ordered, and settable
    uncertain_param_symbols::Union{Nothing, Tuple{Vararg{Symbol}}}
    p_vec::Any
    tunable_pflat::Union{Nothing, TunableP}
    # TODO - explain that this is the look up table for the p container copy
    settable_symbols::Union{Nothing, Tuple{Vararg{Symbol}}}
    simulation_context::Union{Nothing, NamedTuple}

    # Constructor
    function Model(model_def::ModelDefinition, sys::Any)
        # does warmup exist
        warmup_params = get_warmup_params(model_def.parameters)
        warmup = isempty(warmup_params) ? nothing :  warmup_params
        
        # TODO - ideally it be taken care of at the parsing stage
        multiparams = get_array_params(model_def.parameters)

        #Problem and solution are initially empty as they are created during simulation
        new(model_def, sys, nothing, nothing, warmup, nothing, 
        multiparams, nothing,
        nothing, nothing, nothing, nothing, nothing, nothing)

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
    
    model.param_setter!(new_p, p_vec)

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


# helpers
undual(x) = x
undual(x::ForwardDiff.Dual) = ForwardDiff.value(x)


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
                   sampled_uncertain_params=Any,
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

    # TODO - do you still need params? a dictionary kind of special slow case with a warning?

    # copy here in order to avoid losing the types
    # TODO cleanup? 
    # prepare parameters
    # all multiparams have to be the same length, TODO - check if when parsing initially
    multiparams = model.multiparams
    param_len = isempty(multiparams) ? 1 : length(first(values(multiparams)))
    multiparam_values = Vector{Float64}(undef, param_len)

    # TODO - set the multiparam values to the defaults

    all_updatable_params = [sampled_uncertain_params; multiparam_values]

    T = eltype(all_updatable_params)
    p_work = replace(Tunable(), prob.p, T.(model.tunable_pflat.tunable_parameters))

    # update the uncertain parameters with the drawn samples
    if !isempty(sampled_uncertain_params)
        # the these parameters form the first part of the p_vec be design
        # model.param_setter!(p_work, sampled_uncertain_params)
    end

    if !isnothing(model.warmup_settable)
        # apply the inference parameters
        model.param_setter!(p_work, sampled_uncertain_params)

        # initial params are the warm up params
        sol = solve(prob, solver, p=p_work; solver_opts...)

        # p = Plots.plot(sol, ylims=(0,1000))
        # display(p)

        # set u0 for production run
        u0 = sol.u[end]

        # set u0
        # TODO - cache the setter
        states = unknowns(model.sys)
        u0_setter! = setu(model.sys, states)
        # directly work on the unknowns in the tunable p
        u0_float = undual.(u0)
        u0_setter!(p_work[2], u0_float)
    end

    

    opts_prod = solver_opts
    opts_prod = isempty(saveat) ? opts_prod : merge(opts_prod, (saveat=saveat, ))
    opts_prod = isnothing(save_idxs) ? opts_prod : merge(opts_prod, (save_idxs=save_idxs, ))

    results = Vector{SciMLBase.ODESolution}()
    for i in 1:param_len
        # prepare the parameters for the next run        
        for (j, symbol) in enumerate(model.multiparam_symbols)
            multiparam_values[j] = multiparams[symbol][i]
        end

        # apply the inference parameters + multiparams
        # TODO - optimisation of merge here, this could be a standard array in model that we keep updating by index
        model.param_setter!(p_work, [sampled_uncertain_params; multiparam_values])

        sol = solve(prob, solver; p=p_work, opts_prod...)

        # p = Plots.plot(sol, ylims=(0,1000))
        # display(p)
        
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
    uncertain_param_symbols = Vector{Symbol}()

    warmup_map = get_warmup_params(model.model_def.parameters)
    if !isempty(warmup_map)
        @info "Warm up parameters present. Using initial warmup values. "
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
    model.prob = ODEProblem(model.sys, merge(u0,p_map_vars), tspan)
    
    # TODO - switch to a vector of symbols?
    uncertain_Nums = Vector{Num}(undef, length(uncertain_param_symbols))
    for (i, name) in enumerate(uncertain_param_symbols)
        uncertain_Nums[i] = getproperty(model.sys, name)
    end

    model.uncertain_param_symbols =  Tuple(uncertain_param_symbols)

    # prepare the symbols for multiparams
    multiparams_Nums = Vector{Num}(undef, length(model.multiparams))
    # TODO note that order matters
    multiparam_symbols = Tuple(collect(keys(model.multiparams)))
    for (i, symbol) in enumerate(multiparam_symbols)
        multiparams_Nums[i] = getproperty(model.sys, symbol)
    end

    # this the array which will allow us to understand where the settable symbols are in the p container copy
    settable_symbols = (uncertain_param_symbols..., multiparam_symbols...)

    # settable parameters: uncertain..., multiparam...
    settable_params = Tuple([uncertain_Nums; multiparams_Nums])

    model.multiparam_symbols = multiparam_symbols

    # accounting for warm-up variables is a bit more tricky
    # TODO - implement other cases
    # I will start with our simple case where the warm up
    # already exists in one of the settable params
    # TODO - add test cases for warm up parameters not being in multiparams

    # TODO - for now, store which parameters have the warmup, and how they have to be updated, 
    # so a warmup is a tuple [index, value] but it refers reall yto the index in ordered settable prop.p copy container
    
    # TODO - finding warmup ideally is done after being set in model struct
    warmup_settable = Vector{Pair{Int32, Float64}}(undef, length(warmup_map))
    for (i, (warmup_symbol, warmup_value)) in enumerate(warmup_map)
        warmup_index = findfirst(==(warmup_symbol), settable_symbols)
        push!(warmup_settable, warmup_index => warmup_value)
        if !(warmup_symbol in settable_symbols)
            # TODO account for this case
            error("the warmup parameters is not in settable params")
        end
    end
    model.warmup_settable = warmup_settable

    model.param_setter! = setp(model.sys, settable_params)

    model.p_vec = copy(model.prob.p)

    # tunable p container
    tunable_pflat, _, _ = canonicalize(Tunable(), model.prob.p)
    model.tunable_pflat = TunableP(tunable_pflat)

    model.settable_symbols = settable_symbols

    # model.buffer_func = (p) -> remake_buffer(
    #     model.sys, model.prob.p, Dict(zip(uncertain_syms, p))
    # )
    
    model.simulation_context = (
        t_obs = t_obs,
        obs_state_idx = obs_state_idx,
        solver = solver,
        solver_opts = solver_opts)

    @info "Model built and compiled..."


    return nothing
end