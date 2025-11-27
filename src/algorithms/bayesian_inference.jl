using Turing
using Distributions


function run_inference(model::Model, spec::BayesianSpec)

    println("Running Bayesian Inference...")

    # 1. Set up the model
    setup_model_for_inference(model, spec)

    # 2. Get other bits we need for inference

    # 3. Build turing model

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



end