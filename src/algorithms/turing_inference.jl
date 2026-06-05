using Turing
using Distributions
# using DistributionsAD
using DynamicPPL
using SciMLBase: successful_retcode
# using InteractiveUtils


function run_inference(model::Model, spec::TuringSpec)

    println("Running Turing Inference...")

    # 1. Set up the model
    setup_model_for_inference(model, spec)

    # In the test RPA the order the parameters come out from the MTK system is
    # not the same order as what the user puts in. This can lead to come confusion
    # when writing to a file but the buffer function and setter doe not need a specific value it 
    # goes by name.
    
    # TODO
    # multiparams::Union{Nothing, Dict{Symbol, Tuple{Vararg{Float64}}}}
    # uncertain_param_symbols::Union{Nothing, Tuple{Vararg{Symbol}}}
    # settable_params::Union{Nothing, Tuple{Vararg{Num}}}

    # prepare priors for the uncertain parameters

    initial_params = make_initial_params(model, spec)

    # preallocate for the hot loop the results vector
    results_num = isempty(model.multiparam_values) ? 1 : length(model.multiparam_values)
    prealloc_results_vector = Vector{SciMLBase.ODESolution}(undef, results_num)

    priors = arraydist(model.tunable_priors)
    
    # 2. Build turing model
    fit_fcn = fit(model, spec, priors, spec.data; 
        prealloc_results_vector=prealloc_results_vector)
    #fit_fcn = optim_model()

    # Turing.setprogress!(true)
    
    # 3. Run sampling
    chain = sample(
        fit_fcn,
        spec.sampler,
        spec.sampling_method,
        spec.n_samples,
        spec.n_chains;
        progress=true,
        initial_params=initial_params
    )


    # rename the chain draws to the correct variables
    rename_map = Dict(
        Symbol("uncertain_sampled_values[$i]") => model.tunable_symbols[i] for i in eachindex(model.tunable_symbols)
    )
    chain_named = replacenames(chain, rename_map)
    
    return chain_named
end

function make_initial_params(model::Model, spec::TuringSpec)
    initial_params = InitFromParams((
        σ=spec.noise_initial,
        uncertain_sampled_values=collect(model.tunable_initial),
    ))

    return fill(initial_params, spec.n_chains)
end

# -------------------------------------------------------------------------
# Turing model
# -------------------------------------------------------------------------

"""
    _build_turing_model(model, spec, metadata) -> Turing.Model

# Implementation
- Uses model.evaluate_model() for predictions
- Gets priors from metadata
- Builds likelihood from spec
"""
@model function fit(model, spec, uncertain_priors, data; 
    prealloc_results_vector::Union{Vector{SciMLBase.ODESolution}, Nothing}=nothing,
    )

    σ ~ spec.noise_prior
     
    # FIXME - move arraydist to the outside
    uncertain_sampled_values ~ uncertain_priors

    simulation = spec.simulation

    # @code_warntype 
    sols = simulate!(model, simulation.initial_conditions, simulation.tspan;
        # parameters = drawn_params,
        solver=simulation.solver, 
        # dt=spec.dt, 
        saveat=simulation.t_obs, 
        # inference
        solver_opts = simulation.solver_opts,
        sampled_uncertain_params = uncertain_sampled_values,
        prealloc_results_vector = prealloc_results_vector,
        )

    # all solves succeeded
    if any(sol -> !successful_retcode(sol), sols)
        Turing.@addlogprob! -1e10
        return
    end

    # TODO - generalise choosing how to extract the states
    # predicted = vec(vcat(sol[1,:] for sol in sols))
    predicted = vcat(sols[1][1,:], sols[2][1,:], sols[3][1,:])

    # empty the results for the next run
    # empty!(sols)

        # 2. predicted/data are vectors of same length
    if !(predicted isa AbstractVector)
        println("predicted is not abstract")
        #Turing.@addlogprob! -Inf
        Turing.@addlogprob! -1e10
        return
    end

    if length(predicted) != length(data)
        println("different lengths")
        @show size(predicted) size(data) length(predicted) length(data)
        Turing.@addlogprob! -1e10
        return
    end

    # finite values only
    if !all(isfinite, predicted) || !isfinite(σ) || σ <= 0
        println("not finite")
        Turing.@addlogprob! -1e10
        return
    end    

    data ~ MvNormal(predicted, σ^2 * I)
end
    
