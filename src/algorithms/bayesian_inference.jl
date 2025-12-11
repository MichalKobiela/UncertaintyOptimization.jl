using Turing
using Distributions
using DynamicPPL


function run_inference(model::Model, spec::BayesianSpec)

    println("Running Bayesian Inference...")

    # 1. Set up the model
    setup_model_for_inference(model, spec)
    priors = make_priors(model)
    data = spec.data
    prob = model.prob
    setter = model.param_setter
    buffer_fcn = model.buffer_func
    noise_prior = spec.noise_prior
    uncertain_params = model.uncertain_params
    # In the test RPA the order the parameters come out from the MTK system is
    # not the same order as what the user puts in. This can lead to come confusion
    # when writing to a file but the buffer function and setter doe not need a specific value it 
    # goes by name.
    param_symbols = [ModelingToolkit.getname(p) for p in uncertain_params]
    # 2. Build turing model
    fit_fcn = fit(data, prob,  param_symbols, buffer_fcn, setter, priors, noise_prior, spec)
    #fit_fcn = optim_model()
    
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
    
    @model function fit(data, prob,  param_symbols, buffer_fcn, param_setter, priors, noise_prior, spec)
        
        σ ~ noise_prior
        
        param_values = []
        for sym in param_symbols
            val ~ NamedDist(priors[sym], @varname($sym))
            push!(param_values, val)
        end
        
        p_vec = convert(Vector{eltype(param_values)}, param_values)
 
        # Each MCMC iteration creates its own parameter buffer
        # This is thread-safe because new_p is local to this call
        new_p = buffer_fcn(p_vec)
        param_setter(new_p, p_vec)
        prob_tmp = remake(prob; p=new_p)
        
        # Solve with the local problem
        predicted = Array(solve(prob_tmp, spec.solver; 
                               dt=spec.dt, 
                               saveat=spec.t_obs, 
                               save_idxs=spec.obs_state_idx))
        
        data ~ MvNormal(predicted, σ^2 * I(length(data)))
        
    end
    