using Turing
using Distributions


function run_inference(model::Model, spec::BayesianSpec)

    println("Running Bayesian Inference...")

    # 1. Set up the model
    setup_model_for_inference(model, spec)


    # 2. Build turing model

end

# -------------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------------

# helper to make a distribution object - currently only uniform supported but can extend to others
function make_prior(prior::Dict)

    dist = lowercase(prior["distribution"])
    if dist == "uniform"
        return Uniform(prior["lower"], prior["upper"])
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
- Uses model.run_simulation() for predictions
- Gets priors from metadata
- Builds likelihood from spec
"""

function _build_turing_model(model::Model, spec::BayesianSpec)

    uncertain_params = get_uncertain_parameters(model)
    priors = make_priors(model)

    @model function turing_model(spec, model)

        σ ~ spec.noise_prior
        p_vec = Vector{Float64}(undef, length(uncertain_params))

        for (i, name) in enumerate(uncertain_params)
            p_vec[i] ~ priors[name]
        end

        predicted = evaluate_model(model, p_vec)
        spec.data ~ MvNormal(predicted, σ^2 * I(length(predicted)))

    end

        return turing_model
end