using Turing
using Distributions
# using DistributionsAD
using DynamicPPL
using SciMLBase: successful_retcode
# using InteractiveUtils


function run_inference(model::Model, spec::TuringSpec)

    @info "Running Turing inference"

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
    observed_data = vec(spec.data)
    _validate_observation_layout(observed_data, spec.simulation, results_num)

    prealloc_results_vector = Vector{SciMLBase.ODESolution}(undef, results_num)

    # 2. Build turing model
    fit_fcn = fit(model, spec, model.tunable_priors, observed_data;
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
    validate_initial_tunables(
        model.tunable_symbols,
        model.tunable_initial,
        make_priors(model),
    )

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
        save_idxs = observed_state_save_idxs(model.sys, simulation),
        sampled_uncertain_params = uncertain_sampled_values,
        prealloc_results_vector = prealloc_results_vector,
        )

    # all solves succeeded
    if any(sol -> !successful_retcode(sol), sols)
        Turing.@addlogprob! -1e10
        return
    end

    predicted = _predicted_observations(sols, simulation)

    # empty the results for the next run
    # empty!(sols)

        # 2. predicted/data are vectors of same length
    if !(predicted isa AbstractVector)
        @debug "Predicted observations are not an AbstractVector" predicted_type=typeof(predicted)
        #Turing.@addlogprob! -Inf
        Turing.@addlogprob! -1e10
        return
    end

    if length(predicted) != length(data)
        @debug "Predicted observations and data have different lengths" predicted_size=size(predicted) data_size=size(data) predicted_length=length(predicted) data_length=length(data)
        Turing.@addlogprob! -1e10
        return
    end

    # finite values only
    if !all(isfinite, predicted) || !isfinite(σ) || σ <= 0
        @debug "Predicted observations or noise scale are invalid" all_predicted_finite=all(isfinite, predicted) sigma=σ
        Turing.@addlogprob! -1e10
        return
    end    

    data ~ MvNormal(predicted, σ^2 * I)
end

function _predicted_observations(sols, simulation::SimulationSpec)
    if isempty(sols)
        return Float64[]
    end

    saved_state_positions = 1:observed_state_count(simulation)
    return reduce(vcat, (vec(sol[state_position, :]) for sol in sols for state_position in saved_state_positions))
end

function _validate_observation_layout(data, simulation::SimulationSpec, n_solutions::Integer)
    block_length = length(simulation.t_obs) * observed_state_count(simulation)

    if length(data) % block_length != 0
        error(
            "Inference data length $(length(data)) does not match SimulationSpec: " *
            "expected a multiple of $block_length " *
            "($(length(simulation.t_obs)) time points * $(observed_state_count(simulation)) observed state(s))."
        )
    end

    data_solution_count = div(length(data), block_length)
    if data_solution_count != n_solutions
        error(
            "Inference data contains $data_solution_count solution block(s), " *
            "but the simulation will produce $n_solutions. Check SimulationSpec.obs_state " *
            "and model multiparameter values."
        )
    end

    return nothing
end
    
