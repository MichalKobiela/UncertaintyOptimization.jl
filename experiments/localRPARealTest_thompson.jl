#=
For each posterior sample, the script performs a 1D grid 
search over a scaling factor for one parameter, 
runs a warmup simulation, then evaluates how close 
the warmup state and final post-warmup 
prediction are to a target value.
=#
using Revise
using UncertaintyOptimization
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D;
using OrdinaryDiffEq
using CSV, Tables
using Turing
using SciMLBase: VectorOfArray
using SymbolicIndexingInterface
using Random
using Serialization
using CSV, Tables
using Plots
using DataFrames
using Profile
using StatsPlots
# using SciMLSensitivity


Random.seed!(0);


RPA_model = load_model("./test/test-data/RPA_real/opt.yml")
@mtkcompile sys = System(RPA_model.equations, t)
model = Model(RPA_model, sys)

time = CSV.File(joinpath(@__DIR__, "RPA_real_data/time_points.csv")).time

# load the precomputed posterior from a single chain for now
posterior = open(string(@__DIR__)*"/reference/rpareal_chain_reference.jls", "r") do io
        # extract and convert into a DataFrame
        posterior_sample = deserialize(io)[1000:2000]
        DataFrame(posterior_sample)
end


# thompson sampling, grid search loss function
# for each posterior sample check what is the best kx2 scaling factor
# now these will be reserved keywords, 
# so warmup_sol and predicted_sol will contain the data necessary  
function loss(warmup_sol, predicted_sol; sys=nothing)
    # define the loss function using the outputs from warmup and predicted

    isnothing(sys) && error("loss requires the model system as `sys` to locate state A.")

    A = getproperty(sys, :A)
    state_order = unknowns(sys)
    A_idx = findfirst(isequal(A), state_order)
    isnothing(A_idx) && error("State A was not found in the model unknown order.")

    background_fluorescence = 17.6
    sol_end_state(sol, state_idx) = begin
        values = Array(sol)
        if ndims(values) == 1
            state_idx == 1 || error("Solution only contains one saved state, but requested state index $state_idx.")
            return values[end]
        end

        return values[state_idx, end]
    end

    adjusted_predicted = sol_end_state(predicted_sol, A_idx) + background_fluorescence
    warmup = sol_end_state(warmup_sol, A_idx) + background_fluorescence
    
    target = 50
    (((warmup - target).^2) + (adjusted_predicted - target).^2) / 2
end


sim_spec = SimulationSpec(
    t_obs = time,
    # TODO - ideally this would not be operating on idx, but on names
    obs_state_idx = 1,
    initial_conditions = (24.0, 350.0),
    tspan = (0.0, 10.0),
    solver = Rosenbrock23(),
    solver_opts = (dtmin = 1e-12,),
)


# scale - rename to "compute all thompson samples" 
scan = GridScan(
    simulation = sim_spec,
    scale = :kx2,
    linrange = LinRange(0.01, 3, 100),
    lossf = loss,
)
chain = run_scan(posterior, scan, model)
