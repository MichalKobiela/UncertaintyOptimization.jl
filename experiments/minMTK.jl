using Revise
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


Random.seed!(0);


using ModelingToolkit

# Define a nonlinear system
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
@parameters kcymRtot [tunable = false]
@parameters kx2 [tunable = true]
@parameters nx2 [tunable = true]
@parameters kr [tunable = true]
@parameters nr [tunable = true]
@parameters r1 [tunable = true]
@parameters r2 [tunable = true]
@parameters kx3 [tunable = false]
@parameters cuma [tunable = false]

eqs = [
    D(B) ~ 0.02 * (alpha_3 * (((B + sqrt(B^2 + 1e-12)) / 2)^nx2) / (kx3^nx2 + ((B + sqrt(B^2 + 1e-12)) / 2)^nx2) + beta_3) *
          (alpha_4 / (1 + (((A + sqrt(A^2 + 1e-12)) / 2) / kr)^nr) + beta_4) -
          0.1 * r2 * ((B + sqrt(B^2 + 1e-12)) / 2),
    D(A) ~ 0.02 * (alpha_1 / (1 + (kcymRtot / (1 + cuma / kx1))^nx1) + beta_1) *
          (alpha_2 * (((B + sqrt(B^2 + 1e-12)) / 2)^nx2) / (kx2^nx2 + ((B + sqrt(B^2 + 1e-12)) / 2)^nx2) + beta_2) -
          0.02 * r1 * ((A + sqrt(A^2 + 1e-12)) / 2)
       
        ]
@mtkcompile ns = System(eqs, t)

guesses = [
    alpha_1 => 83.4743,
    alpha_2 => 391.1627,
    alpha_3 => 17.7437,
    alpha_4 => 8.7519e6,
    beta_1  => 11.9586,
    beta_2  => 3.9e-4,
    beta_3  => 0.6644,
    beta_4  => 7.1347,
    kx1     => 1.28e-8,
    nx1     => 2.34,
    kx2     => 36.4063,
    nx2     => 1.3,
    kr      => 0.51,
    nr      => 3.2,
    r1      => 89.0635,
    r2      => 7.0188,
]

ps = [kx3 => 4006.9, kcymRtot => 2.75e3, cuma => 2e-6]

u0 = [A => 24.0, B => 350.0]
p = vcat(guesses, ps)
prob = ODEProblem(ns, vcat(u0, p), (0.0, 10.0))

# warm = solve(prob, Rosenbrock23())
# # display(Plots.plot(sol))

# prob_1 = remake(prob; p = Dict(:cuma => 2e-5), u0=warm[end])
# sol = solve(prob_1, Rosenbrock23())
# # display(Plots.plot(sol))

# prob_1 = remake(prob; p = Dict(:cuma => 0.001), u0=warm[end])
# sol = solve(prob_1, Rosenbrock23())
# display(Plots.plot(sol))

@model function fit(data::AbstractVector, prob, saveat::AbstractVector)

    σ ~ InverseGamma(2, 3)

    alpha_1 ~ Distributions.Uniform(0.0, 2000.0)
    alpha_2 ~ Distributions.Uniform(0.0, 250.0)
    alpha_3 ~ Distributions.Uniform(0.0, 1e4)
    alpha_4 ~ Distributions.Uniform(0.0, 1e13)
    beta_1 ~ Distributions.Uniform(0.0, 200.0)
    beta_2 ~ Distributions.Uniform(0.0, 100.0)
    beta_3 ~ Distributions.Uniform(0.0, 5e3)
    beta_4 ~ Distributions.Uniform(0.0, 5000.0)
    kx1 ~ Distributions.Uniform(0.0, 3e-8)
    nx1 ~ Distributions.Uniform(0.0, 5.0)
    kx2 ~ Distributions.Uniform(0.0, 1e4)
    nx2 ~ Distributions.Uniform(1.0, 10.0)
    kr ~ Distributions.Uniform(0.0, 100.0)
    nr ~ Distributions.Uniform(1.0, 100.0)
    r1 ~ Distributions.Uniform(0.0, 1000.0)
    r2 ~ Distributions.Uniform(0.0, 1000.0)

    param_dict = Dict{Symbol, Number}(
        :alpha_1 => alpha_1,
        :alpha_2 => alpha_2,
        :alpha_3 => alpha_3,
        :alpha_4 => alpha_4,
        :beta_1  => beta_1,
        :beta_2  => beta_2,
        :beta_3  => beta_3,
        :beta_4  => beta_4,
        :kx1     => kx1,
        :nx1     => nx1,
        :kx2     => kx2,
        :nx2     => nx2,
        :kr      => kr,
        :nr      => nr,
        :r1      => r1,
        :r2      => r2,
        :cuma => 2e-5 # manual
    )

    solve_opts = (dtmin=1e-12, saveat=saveat)
    
    # Solve the ODE
    try
        # testing
        # param_dict = guesses

        # warm up
        prob = remake(prob; p = param_dict)
        warm = solve(prob, Rosenbrock23())
        # display(Plots.plot(warm))
        
        cdict = Dict(:cuma => 2e-5)
        prob = remake(prob; p = cdict, u0=warm.u[end])
        sol1 = solve(prob, Rosenbrock23(); solve_opts...)
        # display(Plots.plot(sol1))

        cdict[:cuma] = 0.0001
        prob = remake(prob; p = cdict, u0=warm.u[end])
        sol2 = solve(prob, Rosenbrock23(); solve_opts...)

        cdict[:cuma] = 0.001
        prob = remake(prob; p = cdict, u0=warm.u[end])
        sol3 = solve(prob, Rosenbrock23(); solve_opts...)
        # display(Plots.plot(sol3))
        
        data ~ MvNormal(vcat(sol1[1,:], sol2[1,:], sol3[1,:]), σ^2 * I)
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


model2 = fit(data_subset, prob, time)

# Initilize parameters using results from the RPA paper
init_params = Dict(
    :σ => 3.0,
    :alpha_1 => 83.4743,
    :kx1 => 1.28e-8,
    :nx1 => 2.34,
    :beta_1 => 11.9586,
    :alpha_2 => 391.1627,
    :kx2 => 36.4063,
    :nx2 => 1.3,
    :beta_2 => 3.9e-4,
    :alpha_4 => 8.7519e6,
    :kr => 0.51,
    :nr => 3.2,
    :beta_4 => 7.1347,
    :r1 => 89.0635,
    :r2 => 7.0188,
    :alpha_3 => 17.7437,
    :beta_3 => 0.6644
)

# init_params_arr = [3.0,83.4743, 1.28e-8, 2.34, 2.75e3, 11.9586, 391.1627, 36.4063, 1.3, 3.9e-4, 8.7519e6, 0.51, 3.2, 7.1347, 89.0635, 7.0188, 17.7437, 4006.9, 0.6644]

Random.seed!(4)
nuts = NUTS(0.65,init_ϵ = 0.001)

chain_1 = sample(model2, nuts , MCMCSerial(), 3000, 1, init_params = init_params)

f = open(string(@__DIR__)*"/minmtk_c2_corrected_u0_remake.jls", "w")
serialize(f, chain_1)
close(f)
