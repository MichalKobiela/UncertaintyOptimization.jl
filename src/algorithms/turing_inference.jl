using Turing
using Distributions
using DynamicPPL
using SciMLBase: successful_retcode


function run_inference(model::Model, spec::TuringSpec)

    println("Running Turing Inference...")

    # 1. Set up the model
    setup_model_for_inference(model, spec)
    priors = make_priors(model)
    # In the test RPA the order the parameters come out from the MTK system is
    # not the same order as what the user puts in. This can lead to come confusion
    # when writing to a file but the buffer function and setter doe not need a specific value it 
    # goes by name.
    param_symbols = [ModelingToolkit.getname(p) for p in model.uncertain_params]
    # 2. Build turing model
    fit_fcn = fit(model, spec, param_symbols, priors)
    #fit_fcn = optim_model()

    Turing.setprogress!(true)

    @show spec
    
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
function make_priors(model::Model)
    priors = Dict{Symbol, Distribution}()

    for (name, ps) in model.model_def.parameters
        if ps.role == :uncertain
            priors[name] = make_prior(ps.prior)
        end
    end

    return priors
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
@model function fit(model, spec, param_symbols, priors)

    σ ~ spec.noise_prior
    
    drawn_params = Dict()
    for sym in param_symbols
        val ~ NamedDist(priors[sym], sym)
        drawn_params[sym] = val
    end

    sols = simulate!(model, spec.initial_conditions, spec.tspan;
        parameters = drawn_params,
        solver=spec.solver, 
        # dt=spec.dt, 
        saveat=spec.t_obs, 
        # inference
        solver_opts = spec.solver_opts,
        save_idxs=spec.obs_state_idx
        )


    # treat non-successful terminations / NaNs as impossible
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
    