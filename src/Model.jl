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
    
    # Constructor
    function Model(model_def::ModelDefinition, sys::Any)
        new(model_def, sys, nothing, 
        # tunables
        nothing, () :: Tuple{Vararg{Float64}}, () :: Tuple{Vararg{Float64}},
        # warmup
        nothing, () :: Tuple{Vararg{Float64}},
        # multiparams
        nothing, () :: Tuple{Vararg{Float64}})
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

function extract_multiparams(parameters::Dict{Symbol, ParameterSpec})::Dict{Symbol, Tuple{Vararg{Float64}}}
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
                   prealloc_results_vector::Union{Nothing, Vector{SciMLBase.ODESolution}} = nothing
                   )

    # build the problem once
    if isnothing(model.prob)
        println("set up simulation")
        setup_simulation!(model, 
                        initial_conditions, 
                        tspan;
                        parameters
                        )
    end

    prob = model.prob
    multiparam_length = isempty(model.multiparam_values) ? 1 : length(model.multiparam_values)

    if !isnothing(sampled_uncertain_params)
        p_work = replace(Tunable(), prob.p, sampled_uncertain_params)
    else
        ## NON-INFERENCE Sim
        # a deep copy because we are modifying a cuma parameter
        # ? - we could set the cuma parameter here accordingly (in the warmup stage)
        # this way we won't have to rely on the "original container settings"
        p_work = deepcopy(prob.p)

        # FIXME - consider an inner function depending on the parameters present, but most likely this preallocaiton is redundant
        prealloc_results_vector = Vector{SciMLBase.ODESolution}(undef, multiparam_length)
    end

    if !isempty(model.warmup_values)
        # Reset warmup parameters on every simulation call. Some parameter containers
        # returned by `replace(Tunable(), ...)` share non-tunable storage with `prob.p`,
        # so production-stage multiparameter updates can otherwise leak into the next
        # warmup solve.
        model.warmup_setter!(p_work, model.warmup_values)

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
        if !isempty(model.multiparam_values)
            model.multiparam_setter!(p_work, model.multiparam_values[i])
        end

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

    ordered_params = ModelingToolkit.parameters(model.sys)


    ## warmup
    # TODO - finding warmup ideally is done after being set in model struct
    warmup_settable = Vector{Pair{Int32, Float64}}(undef, length(warmup_map))
    warmup_Nums = Vector{Num}(undef, length(warmup_map))
    # we assume warmup parameters have only one value (no multiple stages)
    warmup_values = Vector{Float64}(undef, length(warmup_map))

    counter = 1
    for p in ordered_params
        s = Symbolics.tosymbol(p)

        if haskey(warmup_map, s)
            warmup_Nums[counter] = getproperty(model.sys, s)
            warmup_values[counter] = warmup_map[s]
            counter += 1
        end
    end
    model.warmup_values = Tuple(warmup_values)
    model.warmup_setter! = setp(model.sys, warmup_Nums)


    ## multiparams
    multiparams = extract_multiparams(model.model_def.parameters)

    # prepare the MTK specific Nums
    multiparams_Nums = Vector{Num}(undef, length(multiparams))
    multiparam_values = Vector{Tuple}(undef, length(multiparams))

    ordered_params = ModelingToolkit.parameters(model.sys)

    # add the multiparameter using the MTK order
    counter = 1
    for p in ordered_params
        s = Symbolics.tosymbol(p)

        if haskey(multiparams, s)
            multiparams_Nums[counter] = getproperty(model.sys, s)
            multiparam_values[counter] = multiparams[s]
            counter += 1
        end
    end

    # we have to flip the values so that the first set contains the first column
    multiparam_values_flipped = [collect(x) for x in zip(multiparam_values...)]
    # extract the actual values and structure them in the right order
    # these will be used later with the psetter
    model.multiparam_values = Tuple(Tuple.(multiparam_values_flipped))

    # setter for multiparams
    model.multiparam_setter! = setp(model.sys, multiparams_Nums)

    ## tunable
    tunable_params = [p for p in ordered_params if p in Set(ModelingToolkit.tunable_parameters(model.sys))]

    model.tunable_symbols = Tuple(Symbolics.tosymbol(p) for p in tunable_params)
    model.tunable_priors = arraydist(make_priors(model))
    model.tunable_initial = get_initial_tunables(model)
    
    @info "Model built and compiled..."

    return nothing
end

# helper to build all priors for all uncertain params
function make_priors(model::Model)::Vector{Uniform{Float64}}
    ordered_params = ModelingToolkit.parameters(model.sys)
    tunable_params = [p for p in ordered_params if p in Set(ModelingToolkit.tunable_parameters(model.sys))]
    
    priors = Vector{Uniform{Float64}}(undef, length(tunable_params))

    for (i, param) in enumerate(tunable_params)
        s = Symbolics.tosymbol(param)

        for (param_symbol, param_spec) in model.model_def.parameters
            if param_symbol == s
                if param_spec.role != :uncertain
                    error("A found uncertain parameter $symbol is not uncertain")
                end

                priors[i] = make_prior(param_spec.prior)
            end
        end
    end

    return priors
end

# helper to make a distribution object - currently only uniform supported but can extend to others
function make_prior(prior::Dict)
    dist = lowercase(prior["distribution"])
    if dist == "uniform"
        return Distributions.Uniform(prior["lower"], prior["upper"])
    end
    
    error("Unsupported prior distribution: $(prior["distribution"])")
end

function get_initial_tunables(model::Model)::Tuple{Vararg{Float64}}
    ordered_params = ModelingToolkit.parameters(model.sys)
    tunable_params = [p for p in ordered_params if p in Set(ModelingToolkit.tunable_parameters(model.sys))]

    initial_tunable_values = Vector{Float64}(undef, length(tunable_params))

    for (i, param) in enumerate(tunable_params)
        s = Symbolics.tosymbol(param)

        for (param_symbol, param_spec) in model.model_def.parameters
            if param_symbol == s
                if isnothing(param_spec.value)
                    error("No initial value found for uncertain parameter $symbol. Provide it in the YAML or via spec.simulation.uncertain_param_values.")
                elseif param_spec.value isa Tuple || param_spec.value isa AbstractArray
                    error("Uncertain parameter $symbol has a non-scalar initial value, which is not supported for Turing initialisation.")
                end

                initial_tunable_values[i] = float(param_spec.value)
            end
        end
    end

    return Tuple(initial_tunable_values)
end