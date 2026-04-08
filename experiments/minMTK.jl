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
warm = solve(prob, Rosenbrock23())
display(Plots.plot(sol))

prob_1 = remake(prob; p = Dict(:cuma => 2e-5), u0=warm[end])
sol = solve(prob_1, Rosenbrock23())
display(Plots.plot(sol))

prob_1 = remake(prob; p = Dict(:cuma => 0.001), u0=warm[end])
sol = solve(prob_1, Rosenbrock23())
display(Plots.plot(sol))

