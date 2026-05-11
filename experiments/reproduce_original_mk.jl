using Revise
# avoid issues of world-age
using ModelingToolkit, RuntimeGeneratedFunctions
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D;
using OrdinaryDiffEq
using CSV, Tables
using Turing
using SciMLBase: VectorOfArray
using SciMLStructures: Tunable, canonicalize, replace, replace!
using SymbolicIndexingInterface
using Random
using Serialization
using CSV, Tables
using Plots
using DataFrames
using Distributions
using Symbolics
# using DistributionsAD
using BenchmarkTools
using ForwardDiff
using ForwardDiff: Dual
using DynamicPPL

Random.seed!(0);

order = [:alpha_1, :kx1, :nx1, :beta_1, :alpha_2, :kx2, :nx2, :beta_2, :alpha_4, :kr, :nr, :beta_4,
    :r1, :r2, :alpha_3, :beta_3, :kx3, :kcymRtot, :cuma]

# the tunable parameters are listed first
tunable_last_idx = 16

tunable_params = order[1: tunable_last_idx]
fixed_params = order[tunable_last_idx+1: end]

function odes_warm_up!(du, y, p, t)
    y = max.(y, 0)
    # Parameters
    alpha_1, kx1, nx1, beta_1, alpha_2, kx2, nx2, beta_2, alpha_4, kr, nr, beta_4,
    r1, r2, alpha_3, beta_3, kx3, kcymRtot, cuma = p
    # ODE equations
    du[1] = 0.02 * (alpha_1 / (1 + (kcymRtot / (1 + cuma / kx1))^nx1) + beta_1) *
            (alpha_2 * (y[2]^nx2) / (kx2^nx2 + y[2]^nx2) + beta_2) - 0.02 * r1 * y[1]

    du[2] = 0.02 * (alpha_3 * (y[2]^nx2) / (kx3^nx2 + y[2]^nx2) + beta_3) *
            (alpha_4 / (1 + (y[1] / kr)^nr) + beta_4) - 0.1 * r2 * y[2]
end


guess_map = Dict{Symbol,Float64}(
    :alpha_1 => 83.4743,
    :alpha_2 => 391.1627, #  20.0, # 391.1627,
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

fixed_params_values = [guess_map[s] for s in fixed_params]

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

tspan = (0.0, 10.0)

params = [guess_map[p] for p in order]

u0 = [24.0, 350.0]
# u0_map = Dict(A => 24.0, B => 350.0)
A_idx = 1

prob = ODEProblem(odes_warm_up!, u0, tspan, params)

cuma_idx = findfirst(isequal(:cuma), order)

## basic validation
# Solve the ODE
# warm = solve(prob, Tsit5(), u0=u0)
# display(Plots.plot(warm))

# params[cuma_idx] = 20 * 1e-6
# sol = solve(prob, Tsit5(), u0 = warm.u[end])
# display(Plots.plot(sol))

# params[cuma_idx] = 1000 * 1e-6
# sol = solve(prob, Tsit5(), u0 = warm.u[end])
# display(Plots.plot(sol))

# init_params_values = [guess_map[p.name] for p in ordered_ps]

distributions = arraydist([prior_map[s] for s in tunable_params])


@model function fit(
    data::AbstractVector, 
    prob, 
    saveat::AbstractVector)

    σ ~ InverseGamma(2, 3)
    
    draws ~ distributions

    # extend draws with fixed params
    T = typeof(draws[1])
    p_work = Vector{T}(undef, length(order))
    p_work[1:length(draws)] .= draws
    p_work[length(draws)+1:end] .= T.(fixed_params_values)

    try
        warm = solve(prob, Rosenbrock23(), p=p_work, u0=u0, dtmin=1e-12)
        warm_u0 = warm[end]

        p_work[cuma_idx] = 2e-5
        sol1 = solve(prob, Rosenbrock23(); p=p_work, u0=warm_u0, dtmin=1e-12, saveat=saveat)

        p_work[cuma_idx] = 0.0001
        sol2 = solve(prob, Rosenbrock23(); p=p_work, u0=warm_u0, dtmin=1e-12, saveat=saveat)

        p_work[cuma_idx] = 0.001
        sol3 = solve(prob, Rosenbrock23(); p=p_work, u0=warm_u0, dtmin=1e-12, saveat=saveat)
        
        data ~ MvNormal(vcat(sol1[A_idx,:], sol2[A_idx,:], sol3[A_idx,:]), σ^2 * I)
    catch e
        print(e)
        Turing.@addlogprob! -1e10
    end

    return nothing
end

time = CSV.read(string(@__DIR__)*"/RPA_real_data/time_points.csv", 
        DataFrame)[!,1]
data = Matrix(CSV.read(string(@__DIR__)*"/RPA_real_data/data.csv", 
        DataFrame))
background_fluorescence = 17.6
data = data .- background_fluorescence
# select specific modelled data
data_subset = vcat(data[:,2], data[:,5], data[:,9])


model2 = fit(data_subset, prob, time)#, force_values=guesses_values)

init_params_draws = Dict(
    :σ => 3.0,
    :draws => [guess_map[s] for s in tunable_params],
)

Random.seed!(4)
nuts = NUTS(0.55,init_ϵ = 0.003)
# adtype = AutoForwardDiff(chunksize = 17)
chain_1 = sample(model2, nuts , MCMCSerial(), 3, 1, init_params = init_params_draws)

# rename the chain 
rename_map = Dict(
    Symbol("draws[$i]") => tunable_params[i]
    for i in eachindex(tunable_params)
)
chain_named = replacenames(chain_1, rename_map)

f = open(string(@__DIR__)*"/reproduce_original_mk2.jls", "w")
serialize(f, chain_named)
close(f)

