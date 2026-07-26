using Revise
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D;
using OrdinaryDiffEq
using CSV, Tables
using Turing
using SciMLBase: VectorOfArray, successful_retcode
using SciMLStructures: Tunable, Initials, canonicalize, replace, replace!
using SymbolicIndexingInterface
using Random
using Serialization
using CSV, Tables
using Plots
using DataFrames
using Distributions
# using DistributionsAD
using Profile


const SOLVER=Rosenbrock23()

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

eqs = [
    D(B) ~ 0.02 * (alpha_3 * (((B + sqrt(B^2 + 1e-12)) / 2)^nx2) / (kx3^nx2 + ((B + sqrt(B^2 + 1e-12)) / 2)^nx2) + beta_3) *
          (alpha_4 / (1 + (((A + sqrt(A^2 + 1e-12)) / 2) / kr)^nr) + beta_4) -
          0.1 * r2 * ((B + sqrt(B^2 + 1e-12)) / 2),
    D(A) ~ 0.02 * (alpha_1 / (1 + (kcymRtot / (1 + cuma / kx1))^nx1) + beta_1) *
          (alpha_2 * (((B + sqrt(B^2 + 1e-12)) / 2)^nx2) / (kx2^nx2 + ((B + sqrt(B^2 + 1e-12)) / 2)^nx2) + beta_2) -
          0.02 * r1 * ((A + sqrt(A^2 + 1e-12)) / 2)
    ]
# eqs = [
#     D(B) ~ 0.02 * (alpha_3 * (max(B, 0)^nx2) / (kx3^nx2 + max(B, 0)^nx2) + beta_3)  *
#           (alpha_4 / (1 + (max(A, 0) / kr)^nr) + beta_4) -
#           0.1 * r2 * max(B, 0),
#     D(A) ~ 0.02 * (alpha_1 / (1 + (kcymRtot / (1 + cuma / kx1))^nx1) + beta_1) *
#         (alpha_2 * (max(B, 0)^nx2) / (kx2^nx2 + max(B, 0)^nx2) + beta_2) -
#         0.02 * r1 * max(A, 0),
#     ]

@mtkcompile ns = System(eqs, t)

# parameters of the system in the correct order
# which we are going to use for all operations
ordered_params = [p for p in parameters(ns)]

guess_map = Dict{Symbol,Float64}(
    :alpha_1 => 83.4743,
    :alpha_2 => 20.0, # 391.1627,
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
# 
init_params = Dict([p => guess_map[p.name] for p in ordered_params])

# try jac=true with simplify=true
prob = ODEProblem(ns, merge(Dict(u0), init_params), (0.0, 10.0), jac=false, simplify=false)

## settable symbols (tunables)
# FIXME - grab tunables programmatically rather than this list,
all_tunables = ModelingToolkit.tunable_parameters(ns)

# Only keep tunables that are actually in the parameter set
tunable_set = Set(ModelingToolkit.tunable_parameters(ns))
tunable_params = [p for p in parameters(ns) if p in tunable_set]

# get the Nums for the setter
multiparams_Nums = Vector{Num}(undef, length(tunable_params))
for (i, param) in enumerate(tunable_params)
    multiparams_Nums[i] = getproperty(ns, param.name)
end

# create a setter for your specific symbol order
uncertain_setter! = setp(ns, multiparams_Nums)

# cuma setter
cuma_setter! = setp(ns, [getproperty(ns, :cuma),])

# prepare the list of distributions for drawing in the right order
tunable_priors = Vector{Distribution}(undef, length(tunable_params))
for (i, param) in enumerate(tunable_params)
    tunable_priors[i] = prior_map[param.name]
end

tunable_priors2 = arraydist([prior_map[p.name] for p in tunable_params])

state_order = unknowns(ns)
A_idx = findfirst(isequal(A), state_order)
B_idx = findfirst(isequal(B), state_order)

# warm = solve(prob, Rosenbrock23())
# # display(Plots.plot(sol))

# prob_1 = remake(prob; p = Dict(:cuma => 2e-5), u0=warm[end])
# sol = solve(prob_1, Rosenbrock23())
# # display(Plots.plot(sol))

# prob_1 = remake(prob; p = Dict(:cuma => 0.001), u0=warm[end])
# sol = solve(prob_1, Rosenbrock23())
# display(Plots.plot(sol))


@model function fit(data::AbstractVector, prob, saveat::AbstractVector, distributions, ig;
    force_values=nothing)

    σ ~ ig

    draws ~ distributions
    p_work = replace(Tunable(), prob.p, draws)
    
    cuma_setter!(p_work, (2e-6, ))
    # FIXME - the dtmin is hardcoded here
    warm = solve(prob, SOLVER, p=p_work, dtmin=1e-12)

    # update the u0
    p_work = replace(Initials(), p_work, warm.u[end])
    
    cuma_setter!(p_work, (2e-5, ))
    sol1 = solve(prob, SOLVER, p=p_work; dtmin=1e-12, saveat=saveat)
    # display(Plots.plot(sol1))

    cuma_setter!(p_work, (0.0001, ))
    sol2 = solve(prob, SOLVER, p=p_work; dtmin=1e-12, saveat=saveat)

    cuma_setter!(p_work, (0.001, ))
    sol3 = solve(prob, SOLVER, p=p_work; dtmin=1e-12, saveat=saveat)
    # display(Plots.plot(sol3))

    # all solves succeeded
    if any(sol -> !successful_retcode(sol), [sol1, sol2, sol3])
        Turing.@addlogprob! -1e10
        return
    end

    predicted = vcat(sol1[A_idx,:], sol2[A_idx,:], sol3[A_idx,:])

    # predicted/data are vectors of same length
    if !(predicted isa AbstractVector)
        println("predicted is not abstract")
        Turing.@addlogprob! -1e10
        return
    end

    if length(predicted) != length(data)
        pritnln("different lengths")
        Turing.@addlogprob! -1e10
        return
    end

    if !all(isfinite, predicted) || !isfinite(σ) || σ <= 0
        println("not finite")
        Turing.@addlogprob! -1e10
        return
    end
        
    data ~ MvNormal(predicted, σ^2 * I)
end

time = CSV.read(string(@__DIR__)*"/RPA_real_data/time_points.csv", DataFrame)[!,1]
data = Matrix(CSV.read(string(@__DIR__)*"/RPA_real_data/data.csv", DataFrame))
background_fluorescence = 17.6
data = data .- background_fluorescence
# select specific modelled data
data_subset = vcat(data[:,2], data[:,5], data[:,9])


model = fit(data_subset, prob, time, tunable_priors2, InverseGamma(2, 3))#, force_values=guesses_values)

# Initilize parameters using results from the RPA paper
init_params_draws = Dict(
    :σ => 3.0,
    :draws => [guess_map[p.name] for p in tunable_params],
)

Random.seed!(4)
nuts = NUTS(0.65,init_ϵ = 0.001)

# Profile.init(; n=10^7, delay=0.001)
# Profile.clear_malloc_data()
# Profile.clear()
# @profile 
chain_1 = sample(model, nuts , MCMCSerial(), 3000, 1, init_params = init_params_draws)
# data, lidict = Profile.retrieve()
# serialize("profile_results_1thread_minmtk.jlprof", (data, lidict))

# rename the chain 
rename_map = Dict(
    Symbol("draws[$i]") => tunable_params[i].name
    for i in eachindex(tunable_params)
)
chain_named = replacenames(chain_1, rename_map)

f = open(string(@__DIR__)*"/minmtk_brokenc30_repeated_noprofiling_SetInitCorrected.jls", "w")
serialize(f, chain_named)
close(f)

