using Turing
using Distributions
using DistributionsAD
using DynamicPPL
using SciMLBase: successful_retcode


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
    
    # 2. Build turing model
    fit_fcn = fit(model, spec, priors)
    #fit_fcn = optim_model()

    Turing.setprogress!(true)
    
    # 3. Run sampling
    chain = sample(
        fit_fcn,
        spec.sampler,
        spec.sampling_method,
        spec.n_samples,
        spec.n_chains;
        progress=true
    )
    
    return chain
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
@model function fit(model, spec, uncertain_priors)

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

    sols = simulate!(model, spec.initial_conditions, spec.tspan;
        # parameters = drawn_params,
        solver=spec.solver, 
        # dt=spec.dt, 
        saveat=spec.t_obs, 
        # inference
        solver_opts = spec.solver_opts,
        sampled_uncertain_params = uncertain_sampled_values,
        )


    # also treat non-successful terminations / NaNs as impossible
    if !successful_retcode(sols[end].retcode) || any(!isfinite, Array(sols[end]))
        Turing.@addlogprob!(-Inf)
        return
    end

    # this was called before, 
    # "which state variables to save"
    # save_idxs=spec.obs_state_idx)

    # TODO - generalise choosing how to extract the states
    # predicted = vec(vcat(sol[1,:] for sol in sols))
    predicted = vcat(sols[1][1,:], sols[2][1,:], sols[3][1,:])

    data = spec.data
    
    data ~ MvNormal(predicted, σ^2 * I)
end
    