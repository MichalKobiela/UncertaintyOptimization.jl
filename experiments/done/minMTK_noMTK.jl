#=
The purpouse of this script to use extract the kernel function from MTK and run it. 

This was a way to see if we can still use MTK to our advantage, 
    but not pay the price for the MTK-related machinery. 

After extracting and benchmarking this code, I found that the kernel is fast, 
but it is not the primary reason why the overall sampling is slow. 
=#

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
@parameters kx2 [tunable = true]
@parameters nx2 [tunable = true]
@parameters kr [tunable = true]
@parameters nr [tunable = true]
@parameters r1 [tunable = true]
@parameters r2 [tunable = true]
@parameters kcymRtot [tunable = false]
@parameters kx3 [tunable = false]
@parameters cuma [tunable = false]

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
        0.02 * r1 * max(A, 0)
]

# @mtkcompile ns = System(eqs, t)
# @named sys = System(eqs, t)
@named sys = System(eqs, t)
sys = structural_simplify(sys)

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

rhss = [eq.rhs for eq in equations(sys)]
sts   = unknowns(sys)
ps_unordered   = parameters(sys)
iv   = ModelingToolkit.get_iv(sys) # Usually 't'

# Only keep tunables that are actually in the parameter set
tunable_set = Set(ModelingToolkit.tunable_parameters(sys))
tunable_params = [p for p in parameters(sys) if p in tunable_set]
fixed_params = [p for p in parameters(sys) if p ∉ tunable_set]

# tunable first, fixed later
ordered_ps = [tunable_params; fixed_params]

rhss = Symbolics.simplify.(rhss)

# Call build_function on the VECTOR of expressions
# This version is guaranteed to return (f_oop, f_ip)
f_oop, f_ip = build_function(rhss, sts, ordered_ps, iv; expression = Val{false}, cse=true)#, force_SA=true)

J = ModelingToolkit.jacobian(rhss, sts)
# J = Symbolics.simplify.(ModelingToolkit.jacobian(Symbolics.simplify.(rhss), sts))
# J = Symbolics.simplify.(J)
J_oop, J_ip = build_function(J, sts, ordered_ps, iv; expression = Val(false), cse=true)

# f = ODEFunction(f_ip, jac = J_ip)
f = ODEFunction{false, SciMLBase.FullSpecialize}(f_oop, jac = J_oop)
# f = ODEFunction(f_ip)

u0_map = Dict(A => 24.0, B => 350.0)

u0 = [u0_map[s] for s in sts]
p0 = [guess_map[p.name] for p in ordered_ps]
prob = ODEProblem{false, SciMLBase.FullSpecialize}(f, u0, (0.0, 10.0), p0)


init_params_values = [guess_map[p.name] for p in ordered_ps]
fixed_params_values = [guess_map[p.name] for p in fixed_params]

# get the Nums for the setter
# multiparams_Nums = Vector{Num}(undef, length(tunable_params))
# for (i, param) in enumerate(tunable_params)
#     multiparams_Nums[i] = getproperty(ns, param.name)
# end

# cuma setter
# cuma_setter! = setp(ns, [getproperty(ns, :cuma),])

# prepare the list of distributions for drawing in the right order
# tunable_priors = Vector{Distribution}(undef, length(tunable_params))
# for (i, param) in enumerate(tunable_params)
#     tunable_priors[i] = prior_map[param.name]
# end

tunable_priors2 = arraydist([prior_map[p.name] for p in tunable_params])

state_order = unknowns(sys)
# `@named sys = System(...)` namespaces accessors such as `sys.A`, but
# `unknowns(sys)` returns the bare symbols.  Index by the bare symbols so the
# no-MTK problem observes the same physical state as the compiled MTK problem.
A_idx = findfirst(isequal(A), state_order)
B_idx = findfirst(isequal(B), state_order)

params = copy(init_params_values)

cuma_idx = findfirst(isequal(cuma), ordered_ps)

# test_duals = ForwardDiff.Dual{ForwardDiff.Tag{DynamicPPL.DynamicPPLTag, Float64}, Float64, 9}[
#     Dual{ForwardDiff.Tag{DynamicPPL.DynamicPPLTag, Float64}}(8471.275872497094,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), # alpha_3
#     Dual{ForwardDiff.Tag{DynamicPPL.DynamicPPLTag, Float64}}(3892.1280967413495,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), # beta_3
#     Dual{ForwardDiff.Tag{DynamicPPL.DynamicPPLTag, Float64}}(38.1521059537882,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), # beta_2
#     Dual{ForwardDiff.Tag{DynamicPPL.DynamicPPLTag, Float64}}(2.403823873189367,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), # nx1
#     Dual{ForwardDiff.Tag{DynamicPPL.DynamicPPLTag, Float64}}(7.082621328112331e12,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), # alpha_4
#     Dual{ForwardDiff.Tag{DynamicPPL.DynamicPPLTag, Float64}}(2.107440364099724,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), # nx2
#     Dual{ForwardDiff.Tag{DynamicPPL.DynamicPPLTag, Float64}}(89.59331280378838,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), # alpha_2
#     Dual{ForwardDiff.Tag{DynamicPPL.DynamicPPLTag, Float64}}(8192.21592651675,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), # kx2
#     Dual{ForwardDiff.Tag{DynamicPPL.DynamicPPLTag, Float64}}(129.10938198655745,45.763219401803426,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), # beta_1
#     Dual{ForwardDiff.Tag{DynamicPPL.DynamicPPLTag, Float64}}(741.4056646388501,0.0,191.72330508027503,0.0,0.0,0.0,0.0,0.0,0.0,0.0), # r1
#     Dual{ForwardDiff.Tag{DynamicPPL.DynamicPPLTag, Float64}}(6.7229240948540025e-9,0.0,0.0,5.216333815348385e-9,0.0,0.0,0.0,0.0,0.0,0.0), # kx1
#     Dual{ForwardDiff.Tag{DynamicPPL.DynamicPPLTag, Float64}}(144.30022394185826,0.0,0.0,0.0,123.47766931218781,0.0,0.0,0.0,0.0,0.0), # r2
#     Dual{ForwardDiff.Tag{DynamicPPL.DynamicPPLTag, Float64}}(20.995283540748407,0.0,0.0,0.0,0.0,16.5872642311842,0.0,0.0,0.0,0.0), # kr
#     Dual{ForwardDiff.Tag{DynamicPPL.DynamicPPLTag, Float64}}(228.78673067134994,0.0,0.0,0.0,0.0,0.0,202.61504660570753,0.0,0.0,0.0), # alpha_1
#     Dual{ForwardDiff.Tag{DynamicPPL.DynamicPPLTag, Float64}}(68.86082154078191,0.0,0.0,0.0,0.0,0.0,0.0,21.344749821692513,0.0,0.0), # nr
#     Dual{ForwardDiff.Tag{DynamicPPL.DynamicPPLTag, Float64}}(947.5046140951508,0.0,0.0,0.0,0.0,0.0,0.0,0.0,767.9516153488307,0.0),# beta_4
# ]
# T = eltype(test_duals)
# push!(test_duals, T(guess_map[:kcymRtot]))
# push!(test_duals, T(guess_map[:cuma]))
# push!(test_duals, T(guess_map[:kx3]))
# results = @benchmark solve(prob, Rosenbrock23(autodiff = false), p=test_duals, u0=u0; dtmin=1e-12)
# show(stdout, MIME"text/plain"(), results)

# warm = solve(prob, Rosenbrock23(), p=params, u0=u0; dtmin=1e-12)
# display(Plots.plot(warm))

# params[cuma_idx] = 2e-5
# prob_1 = remake(prob; p = params, u0=warm[end])
# sol = solve(prob_1, Rosenbrock23())
# display(Plots.plot(sol))

# params[cuma_idx] = 0.001
# prob_1 = remake(prob; p = params, u0=warm[end])
# sol = solve(prob_1, Rosenbrock23())
# display(Plots.plot(sol))


@model function fit(
    data::AbstractVector, 
    prob, 
    saveat::AbstractVector, 
    distributions, ig;
    force_values=nothing)

    σ ~ ig

    draws ~ distributions

    if !isnothing(force_values)
        draws = force_values
    end

    # extend draws with fixed params
    T = typeof(draws[1])
    p_work = Vector{T}(undef, length(ordered_ps))
    p_work[1:length(draws)] .= draws
    p_work[length(draws)+1:end] .= T.(fixed_params_values)

    try
        warm = solve(prob, Rosenbrock23(autodiff = false), p=p_work, u0=u0, dtmin=1e-12, dense=false)
        warm_u0 = warm[end]

        p_work[cuma_idx] = 2e-5
        sol1 = solve(prob, Rosenbrock23(autodiff = false); p=p_work, u0=warm_u0, dtmin=1e-12, saveat=saveat, dense=false)
        # display(Plots.plot(sol1))

        p_work[cuma_idx] = 0.0001
        sol2 = solve(prob, Rosenbrock23(autodiff = false); p=p_work, u0=warm_u0, dtmin=1e-12, saveat=saveat, dense=false)

        p_work[cuma_idx] = 0.001
        sol3 = solve(prob, Rosenbrock23(autodiff = false); p=p_work, u0=warm_u0, dtmin=1e-12, saveat=saveat, dense=false)
        
        data ~ MvNormal(vcat(sol1[A_idx,:], sol2[A_idx,:], sol3[A_idx,:]), σ^2 * I)
    catch e
        # print(e)
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


model2 = fit(data_subset, prob, time, tunable_priors2, InverseGamma(2, 3))#, force_values=guesses_values)

init_params_draws = Dict(
    :σ => 3.0,
    :draws => [guess_map[p.name] for p in tunable_params],
)

Random.seed!(4)
nuts = NUTS(0.55,init_ϵ = 0.003)
# adtype = AutoForwardDiff(chunksize = 17)
chain_1 = sample(model2, nuts , MCMCSerial(), 3000, 1, init_params = init_params_draws)

# rename the chain 
rename_map = Dict(
    Symbol("draws[$i]") => tunable_params[i].name
    for i in eachindex(tunable_params)
)
chain_named = replacenames(chain_1, rename_map)

f = open(string(@__DIR__)*"/minmtk_noMtk_full5_500usDuals_simrhss.jls", "w")
serialize(f, chain_named)
close(f)

