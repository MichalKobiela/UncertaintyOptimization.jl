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
        
        # Create a NEW problem for this iteration
        # prob_base is never mutated
        prob_iter = remake(prob; p=new_p)
        
        # Solve with the local problem
        predicted = Array(solve(prob_iter, Euler(); 
                               dt=spec.dt, 
                               saveat=spec.t_obs, 
                               save_idxs=spec.obs_state_idx))
        
        data ~ MvNormal(predicted, σ^2 * I(length(data)))
        
    end
    
 
    
#    # uncertain_vec = collect(uncertain_params)


#     @model function fit(data, model, spec, priors, noise_prior)
        

#         σ ~ noise_prior

#        # if !@isdefined(__first_call__)
#         #    global __first_call__ = true
#         #    println("TURING: model.uncertain_params = ", model.uncertain_params)
#        #end

#         p_vec = Vector{Float64}(undef, length(model.uncertain_params))

       

        


#    # println(uncertain_vec[1])
#     #    p_vec = Vector{Float64}(undef, length(model.uncertain_params))

# #for (i, param) in enumerate(model.uncertain_params)
#    # sym = ModelingToolkit.getname(param)   # ← works in all MTK versions
#    # p_vec[i] ~ priors[sym]
# #end
        
#         #println("Turing uncertain_param dict keys (order of insertion): ", uncertain_params)
#         #println("Canonical uncertain parameters order (uncertain_vec): ", uncertain_vec)
        
#        # p_vec = [uncertain_vec[name] for name in model.uncertain_params]
#         #println(p_vec)
#        # println(uncertain_vec
#        # println("Canonical parameter order for buffer/setter:")
#        # for (name, val) in zip(model.uncertain_params, p_vec_ordered)
#         #    println("    $name => $val")
#        # end

#         #numeric_vals = first.(p_vec)  # extract real value from Dual numbers
#         # print for debugging orders
#         #println("Correct canonical-order parameter vector:")
#         #for (name, val) in zip(p_vec, numeric_vals)
#          #   println("    $name => $val")
#         #end
        
#         #println(p_vec)
        
#       #  predicted = evaluate_model(model, p_vec)

#     #     p_vec = Vector{Float64}(undef, length(model.uncertain_params))
    
#     # for (i, param) in enumerate(model.uncertain_params)
#     #     sym = ModelingToolkit.getname(param)
        
#     #     # Use @varname to create a properly scoped variable
#     #     var = @varname $(Symbol("p_", i))
#     #     p_vec[i] ~ NamedDist(priors[sym], var)
#     # end
        
#      p_vec = Vector{Real}(undef, length(model.uncertain_params))
        
#         for (i, param) in enumerate(model.uncertain_params)
#             sym = ModelingToolkit.getname(param)
            
#             # CRITICAL: Use the actual parameter name as the Turing variable
#             # This makes the chain have beta_RA, beta_AB, etc.
#             p_vec[i] ~ NamedDist(priors[sym], @varname($sym))
#         end

#       new_p = model.buffer_func(p_vec)
    
#       model.param_setter(new_p, p_vec)

#       prob_new = remake(model.prob; p=new_p)

#       predicted = Array(solve(prob_new, Euler(); dt=0.01, saveat=spec.t_obs, save_idxs=1))
    

#         data ~ MvNormal(predicted, σ^2 * I(length(data)))

#         return nothing
#     end

