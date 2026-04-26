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
    priors = make_priors(model)
    initial_params = make_initial_params(model, spec)

    multiparams = model.multiparams
    # as in, how many multiparam there is
    multiparam_count = isempty(multiparams) ? 1 : length(keys(multiparams))
    # TODO - initialise, and make explicit ordered
    multiparam_values = Vector{Float64}(undef, multiparam_count)
    # initialise the multiparam values to the first value? what about warmup? 
    for (i, symbol) in enumerate(model.multiparam_symbols)
        # TODO check if warm up has these parameters
        # TODO this is no longer necessary as we have a specific setter up now for multiparam
        if symbol in keys(model.warmup_params)
            multiparam_values[i] = model.warmup_params[symbol]
        else
            multiparam_values[i] = multiparams[symbol][1]
        end
    end

    # ie how many different parameters (experiments) are being compared
    multiparam_length = isempty(multiparams) ? 1 : length(last(first(multiparams)))

    # preallocate for the hot loop the results vector
    prealloc_results_vector = Vector{SciMLBase.ODESolution}(undef, multiparam_length)
    
    # 2. Build turing model
    fit_fcn = fit(model, spec, priors, spec.data; 
        multiparam_values=multiparam_values, 
        multiparam_length = multiparam_length,
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
        init_params=initial_params
    )


    # rename the chain draws to the correct variables
    rename_map = Dict(
        Symbol("uncertain_sampled_values[$i]") => model.uncertain_param_symbols[i] for i in eachindex(model.uncertain_param_symbols)
    )
    chain_named = replacenames(chain, rename_map)
    
    return chain_named
end

function make_initial_params(model::Model, spec::TuringSpec)::Dict{Symbol, Any}
    uncertain_values = Float64[]

    for symbol in model.uncertain_param_symbols
        if haskey(spec.uncertain_param_values, symbol)
            push!(uncertain_values, float(spec.uncertain_param_values[symbol]))
            continue
        end

        param_spec = model.model_def.parameters[symbol]
        if isnothing(param_spec.value)
            error("No initial value found for uncertain parameter $symbol. Provide it in the YAML or via spec.uncertain_param_values.")
        end
        if param_spec.value isa Tuple || param_spec.value isa AbstractArray
            error("Uncertain parameter $symbol has a non-scalar initial value, which is not supported for Turing initialisation.")
        end

        push!(uncertain_values, float(param_spec.value))
    end

    σ_init = try
        float(mode(spec.noise_prior))
    catch
        try
            float(mean(spec.noise_prior))
        catch
            1.0
        end
    end

    return Dict(
        :σ => σ_init,
        :uncertain_sampled_values => uncertain_values,
    )
end

# -------------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------------

# helper to make a distribution object - currently only uniform supported but can extend to others
function make_prior(prior::Dict)

    dist = lowercase(prior["distribution"])
    if dist == "uniform"
        return truncated(Distributions.Uniform(prior["lower"], prior["upper"]), lower = prior["lower"])
    else
        error("Unsupported prior distribution: $(prior["distribution"])")
    end

end

# helper to build all priors for all uncertain params
function make_priors(model::Model)::Tuple{Vararg{Distribution}}
    priors = Vector{Distribution}(undef, length(model.uncertain_param_symbols))

    for (i, symbol) in enumerate(model.uncertain_param_symbols)
        for (param_symbol, param_spec) in model.model_def.parameters
            if param_symbol == symbol
                if param_spec.role != :uncertain
                    error("A found uncertain parameter $symbol is not uncertain")
                end

                priors[i] = make_prior(param_spec.prior)
            end
        end
    end

    return Tuple(priors)
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
    multiparam_values::Vector{Float64}, 
    multiparam_length::Int = 1,
    prealloc_results_vector::Union{Vector{SciMLBase.ODESolution}, Nothing}=nothing,
    )

    σ ~ spec.noise_prior

    # draw params into a vector
    # the problem is that you have to draw only the uncertain params
    # while we have to fill other params the right way
    # so this complicates the architecture, 
    # we could chain them together and do this part by part, with drawing at the end? 
    

    # ntoe that some params are drawn, but some are set with the multiparameters, 
    # but the multiparameter ones have to happen in the simulate()
    # so maybe we can first set uncertain params, 
    # and the rest of symbols we can set with multiparam
     
    # TODO - consider using TArray{Float64} in order to have an explicit type (AD-friendly?)
    uncertain_sampled_values ~ arraydist(collect(uncertain_priors))

    # param_values = []
    # for sym in uncertain_priors_symbols
    #     val ~ NamedDist(priors[sym], @varname($sym))
    #     push!(param_values, val)
    # end


    # @code_warntype 
    sols = simulate!(model, spec.initial_conditions, spec.tspan;
        # parameters = drawn_params,
        solver=spec.solver, 
        # dt=spec.dt, 
        saveat=spec.t_obs, 
        # inference
        solver_opts = spec.solver_opts,
        sampled_uncertain_params = uncertain_sampled_values,
        multiparam_values = multiparam_values,
        multiparam_length = multiparam_length,
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
    