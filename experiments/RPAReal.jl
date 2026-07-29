using UncertaintyOptimization
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D;
using OrdinaryDiffEq
using CSV, Tables
using Turing
using SciMLBase: VectorOfArray
using SymbolicIndexingInterface: setp
using Random
using Serialization
using Plots
using DataFrames
using AdvancedHMC: DenseEuclideanMetric


# Load model
RPA_model = load_model("./test/test-data/RPA_real/cluster.yml")
@mtkcompile sys = System(RPA_model.equations, t)
model = Model(RPA_model, sys)

# Define simulation parameters
init_cond = (24.0, 350.0) # Initial values for y1 and y2
tspan = ((0.0, 10.0), (0.0, 5.45))
        
# Run simulation
sols = simulate!(model, init_cond, tspan)
# for sol in sols
#     p = Plots.plot(sol)
#     display(p)
# end

# load data
data_frame = CSV.read(
    joinpath(@__DIR__, "reference", "RPA_real_data.csv"), DataFrame; normalizenames=true, stripwhitespace=true,
)
# select specific modelled data
data_selected = vcat(
    data_frame.experession20,
    data_frame.experession100,
    data_frame.expression1000,
) .- 17.6 # adjust for background fluorescence

sim_spec = SimulationSpec(
    t_obs = data_frame.time,
    obs_state = :A,
    initial_conditions = (24.0, 350.0),
    tspan = tspan,
    solver = AutoTsit5(Rosenbrock23(autodiff=false)),
    solver_opts = (dtmin = 1e-9,),
)

turing_spec = TuringSpec(
    simulation = sim_spec,
    data = data_selected,
    noise_prior = InverseGamma(2, 3), 
    noise_initial = 3.0, 
    sampler = NUTS(0.5, init_ϵ = 0.0065, max_depth=8, metricT = DenseEuclideanMetric),
    n_samples = 3000,
    n_chains = 1,
)

Random.seed!(6)

chain = run_inference(model, turing_spec)

open(joinpath(@__DIR__, "mtk_a16.jls"), "w") do f
    serialize(f, chain)
end
