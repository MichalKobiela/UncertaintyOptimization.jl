using Turing
using Distributions
using DynamicPPL

function run_inference(model::Model, spec::BayesianSpec)

    println("Running Bayesian Inference...")

    # 1. Set up the model
    setup_model_for_inference(model, spec)


    # 2. Build turing model
    optim_model = _build_turing_model(model, spec)

    # 3. Run sampling
    chain = sample(
        optim_model(),
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
        return truncated(Uniform(prior["lower"], prior["upper"]), lower = prior["lower"])
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

function _build_turing_model(model::Model, spec::BayesianSpec)

    data = spec.data
    noise_prior = spec.noise_prior
    uncertain_params = get_uncertain_parameters(model)
    priors = make_priors(model)
    
    uncertain_vec = collect(uncertain_params)
    

    @model function turing_model()

        σ ~ noise_prior

        uncertain_param = Dict{Symbol, Float64}()

        for name in uncertain_vec
            uncertain_param[name] ~ priors[name]  
        end
        #println("Turing uncertain_param dict keys (order of insertion): ", uncertain_param)
        #println("Canonical uncertain parameters order (uncertain_vec): ", uncertain_vec)

        p_vec = getindex.(Ref(uncertain_param), uncertain_vec)

        #numeric_vals = first.(p_vec)  # extract real value from Dual numbers
        # print for debugging orders
        #println("Correct canonical-order parameter vector:")
        #for (name, val) in zip(uncertain_vec, numeric_vals)
           # println("    $name => $val")
       # end
        
        predicted = evaluate_model(model, p_vec)

        data ~ MvNormal(predicted, σ^2 * I(length(predicted)))

    end

        return turing_model
end