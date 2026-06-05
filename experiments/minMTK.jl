#=
This script is a minimal implementation of MTK that reproduces the original code. 

The standalone script is important in order to ensure that the number of variables does not change. 
=#
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D;
using OrdinaryDiffEq
using Turing
# using SciMLBase: VectorOfArray
using SciMLStructures: Tunable, canonicalize, replace, replace!, Initials
using SymbolicIndexingInterface: setp
using Random
using Serialization
using CSV, Tables, DataFrames
using Plots
using Distributions
using AdvancedHMC: DenseEuclideanMetric


Random.seed!(0);


const SOLVER = AutoTsit5(Rosenbrock23(autodiff=false))
# const SOLVER = Rosenbrock23(autodiff=AutoForwardDiff())
# const SOLVER = Rosenbrock23(autodiff=AutoReverseDiff(compile=false), concrete_jac = true)
# const SOLVER = Rosenbrock23(autodiff=false)

# const SOLVER = AutoTsit5(
#     Rosenbrock23(autodiff=AutoForwardDiff());
#     stiffalgfirst = true,
#     maxstiffstep = 1,
#     maxnonstiffstep = 1000,
#     nonstifftol = 0.1,
#     stifftol = 0.1,
#     dtfac = 1,
# )

@variables A(t) B(t) 
@parameters alpha_1 [tunable = true]
@parameters alpha_2 [tunable = true]
@parameters alpha_3 [tunable = true]
@parameters alpha_4 [tunable = true]
@parameters beta_1 [tunable = true]
@parameters beta_2 [tunable = true]
@parameters beta_3 [tunable = true]
@parameters beta_4 [tunable = true]
@parameters kx1 [tunable = true]
@parameters nx1 [tunable = true]
@parameters kx2 [tunable = true]
@parameters nx2 [tunable = true]
@parameters kr [tunable = true]
@parameters nr [tunable = true]
@parameters r1 [tunable = true]
@parameters r2 [tunable = true]
@parameters kcymRtot [tunable = false]
@parameters kx3 [tunable = false]
@parameters cuma [tunable = false]

# soft
# eqs = [
#     D(B) ~ 0.02 * (alpha_3 * (((B + sqrt(B^2 + 1e-12)) / 2)^nx2) / (kx3^nx2 + ((B + sqrt(B^2 + 1e-12)) / 2)^nx2) + beta_3) *
#           (alpha_4 / (1 + (((A + sqrt(A^2 + 1e-12)) / 2) / kr)^nr) + beta_4) -
#           0.1 * r2 * ((B + sqrt(B^2 + 1e-12)) / 2),
#     D(A) ~ 0.02 * (alpha_1 / (1 + (kcymRtot / (1 + cuma / kx1))^nx1) + beta_1) *
#           (alpha_2 * (((B + sqrt(B^2 + 1e-12)) / 2)^nx2) / (kx2^nx2 + ((B + sqrt(B^2 + 1e-12)) / 2)^nx2) + beta_2) -
#           0.02 * r1 * ((A + sqrt(A^2 + 1e-12)) / 2)
#     ]
eqs = [
    D(B) ~ 0.02 * (alpha_3 * (max(B, 0)^nx2) / (kx3^nx2 + max(B, 0)^nx2) + beta_3)  *
          (alpha_4 / (1 + (max(A, 0) / kr)^nr) + beta_4) -
          0.1 * r2 * max(B, 0),
    D(A) ~ 0.02 * (alpha_1 / (1 + (kcymRtot / (1 + cuma / kx1))^nx1) + beta_1) *
        (alpha_2 * (max(B, 0)^nx2) / (kx2^nx2 + max(B, 0)^nx2) + beta_2) -
        0.02 * r1 * max(A, 0),
    ]

@mtkcompile ns = System(eqs, t)

ordered_params = [p for p in parameters(ns)]

guess_map = Dict{Symbol,Float64}(
    :alpha_1 => 83.4743,
    :alpha_2 => 25.0, # 391.1627,
    :alpha_3 => 17.7437,
    :alpha_4 => 8.7519e6,
    :beta_1  => 11.9586,
    :beta_2  => 3.9e-4,
    :beta_3  => 0.6644,
    :beta_4  => 7.1347,
    :kx1     => 1.28e-8,
    :nx1     => 2.34,
    :kx2     => 36.4063,
    :nx2     => 1.3,
    :kr      => 0.51,
    :nr      => 3.2,
    :r1      => 89.0635,
    :r2      => 7.0188,
    :kcymRtot => 2.75e3, 
    :kx3 => 4006.9, 
    :cuma => 2e-6
)

prior_map = Dict{Symbol,Distribution}(
    :alpha_1 => Uniform(0.0, 2000.0),
    :alpha_2 => Uniform(0.0, 250.0),
    :alpha_3 => Uniform(0.0, 1e4),
    :alpha_4 => Uniform(0.0, 1e13),
    :beta_1  => Uniform(0.0, 200.0),
    :beta_2  => Uniform(0.0, 100.0),
    :beta_3  => Uniform(0.0, 5e3),
    :beta_4  => Uniform(0.0, 5000.0),
    :kx1     => Uniform(0.0, 3e-8),
    :nx1     => Uniform(1.0, 5.0),
    :kx2     => Uniform(0.0, 1e4),
    :nx2     => Uniform(1.0, 10.0),
    :kr      => Uniform(0.0, 100.0),
    :nr      => Uniform(1.0, 100.0),
    :r1      => Uniform(0.0, 1000.0),
    :r2      => Uniform(0.0, 1000.0),
)

u0 = [A => 24.0, B => 350.0]
initial_params = Dict([p => guess_map[p.name] for p in ordered_params])
prob = ODEProblem(ns, merge(Dict(u0), initial_params), (0.0, 10.0), jac=true, simplify=false)

# prepare cuma setter and priors
tunable_params = [p for p in ordered_params if p in Set(ModelingToolkit.tunable_parameters(ns))]
tunable_priors = arraydist([prior_map[p.name] for p in tunable_params])

cuma_setter! = setp(ns, [getproperty(ns, :cuma),])

# find the states
state_order = unknowns(ns)
A_idx = findfirst(isequal(ns.A), state_order)
B_idx = findfirst(isequal(ns.B), state_order)

# warm = solve(prob, Rosenbrock23())
# # display(Plots.plot(sol))

# prob_1 = remake(prob; p = Dict(:cuma => 2e-5), u0=warm[end])
# sol = solve(prob_1, Rosenbrock23())
# # display(Plots.plot(sol))

# prob_1 = remake(prob; p = Dict(:cuma => 0.001), u0=warm[end])
# sol = solve(prob_1, Rosenbrock23())
# display(Plots.plot(sol))


@model function fit(data::AbstractVector, prob, saveat::AbstractVector, distributions, noise_prior)

    σ ~ noise_prior

    draws ~ distributions
    p_work = replace(Tunable(), prob.p, draws)

    # Solve the ODE
    try
        cuma_setter!(p_work, (2e-6, ))
        warm = solve(prob, SOLVER, p=p_work, dtmin=1e-12)

        p_work = replace(Initials(), p_work, warm.u[end])
        
        cuma_setter!(p_work, (2e-5, ))
        sol1 = solve(prob, SOLVER, p=p_work; dtmin=1e-12, saveat=saveat)

        cuma_setter!(p_work, (0.0001, ))
        sol2 = solve(prob, SOLVER, p=p_work; dtmin=1e-12, saveat=saveat)

        cuma_setter!(p_work, (0.001, ))
        sol3 = solve(prob, SOLVER, p=p_work; dtmin=1e-12, saveat=saveat)
        
        data ~ MvNormal(vcat(sol1[A_idx,:], sol2[A_idx,:], sol3[A_idx,:]), σ^2 * I)
    catch e
        # print(e)
        Turing.@addlogprob! -1e10
    end

    return nothing
end

# prepare data (time point and measurements)
time = CSV.File(joinpath(@__DIR__, "RPA_real_data/time_points.csv")).time
data = Matrix(CSV.read(string(@__DIR__)*"/RPA_real_data/data.csv", DataFrame))
background_fluorescence = 17.6
data = data .- background_fluorescence
# select specific experimental data conditions
data_subset = vcat(data[:,2], data[:,5], data[:,9])


model = fit(data_subset, prob, time, tunable_priors, InverseGamma(2, 3))

initial_params_draws = (;
    σ = 3.0,
    draws = [guess_map[p.name] for p in tunable_params],
)

Random.seed!(4)
sampler = NUTS(0.5,init_ϵ = 0.005, metricT = DenseEuclideanMetric)
chain_1 = sample(model, sampler , MCMCSerial(), 3, 1, initial_params = [InitFromParams(initial_params_draws)])

rename_map = Dict(
    Symbol("draws[$i]") => tunable_params[i].name
    for i in eachindex(tunable_params)
)
chain_named = replacenames(chain_1, rename_map)

f = open(string(@__DIR__)*"/minmtk_r11_correctInitials_test.jls", "w")
serialize(f, chain_named)
close(f)
