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
# using ProfileView
# using StatProfilerHTML
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
function loss(warmup_sol, predicted_sol)
    # define the loss function using the outputs from warmup and predicted

    # we previously only saved one state, 
    # FIXME - show how to find the variable
    stateA = 1

    background_fluorescence = 17.6
    adjusted_predicted = predicted_sol[stateA, end] + background_fluorescence
    warmup = warmup_sol[stateA, end] + background_fluorescence
    
    target = 50
    (((warmup - target).^2) + (adjusted_predicted - target).^2) / 2
end


# so this is what will go into a grid spec
# and we will have to pass the loss funtion to it? 
# so parameters:
# - posterior
# - grid
# - loss function
function reallynothere()
    thompson_samples = []
    for row in eachrow(posterior)
        # the fixed values should come from the setup, 
        # whereas the tunables should be used for evaluation

        # optimize loss
        values = collect(LinRange(0.01, 3, 100))
        loss_post = x -> loss(row, x)
        losses = loss_post.(values)
        min_loss_index = argmin(losses)

        push!(thompson_samples, values[min_loss_index])
    end
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
    scale = "kx2",
    linrange = LinRange(0.01, 3, 100),
    lossf = loss,
)


chain = run_scan(posterior, scan)