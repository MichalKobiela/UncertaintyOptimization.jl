using Revise
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
using DistributionsAD
# using ReverseDiff, Memoization

# Turing.setadbackend(:reversediff)
# Turing.setrdcache(true) # This is the magic "make it fast" button for MTK


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

# Instead of @mtkcompile or structural_simplify(sys)
# Generate the actual Julia code for the ODE
# f_expr = build_function(sys, states(sys), parameters(sys), t; expression=Val{false})

# f_expr[2] is the in-place version: f(du, u, p, t)
# prob = ODEProblem(f_expr[2], u0, tspan, ps)


# parameters of the system in the correct order
# which we are going to use for all operations
ordered_params = [p for p in parameters(ns)]

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

# create a tunable
tunable_ps, _, _ = canonicalize(Tunable(), prob.p)
# FIXME - check the order after canonicalize
# it should be the same as manual list of parameters

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

    # draw uncertain
    # alpha_1 ~ Distributions.Uniform(0.0, 2000.0)
    # alpha_2 ~ Distributions.Uniform(0.0, 250.0)
    # alpha_3 ~ Distributions.Uniform(0.0, 1e4)
    # alpha_4 ~ Distributions.Uniform(0.0, 1e13)
    # beta_1 ~ Distributions.Uniform(0.0, 200.0)
    # beta_2 ~ Distributions.Uniform(0.0, 100.0)
    # beta_3 ~ Distributions.Uniform(0.0, 5e3)
    # beta_4 ~ Distributions.Uniform(0.0, 5000.0)
    # kx1 ~ Distributions.Uniform(0.0, 3e-8)
    # nx1 ~ Distributions.Uniform(1.0, 5.0)
    # kx2 ~ Distributions.Uniform(0.0, 1e4)
    # nx2 ~ Distributions.Uniform(1.0, 10.0)
    # kr ~ Distributions.Uniform(0.0, 100.0)
    # nr ~ Distributions.Uniform(1.0, 100.0)
    # r1 ~ Distributions.Uniform(0.0, 1000.0)
    # r2 ~ Distributions.Uniform(0.0, 1000.0)

    draws ~ distributions

    if !isnothing(force_values)
        draws = force_values
    end

    # prepare container for the new types
    T = eltype(draws)

    # TODO - do a test if p_work initially always reflects on prob.p, or if cuma can leak
    right_types = T.(tunable_ps)
    # @show right_types
    p_work = replace(Tunable(), prob.p, right_types)
    # TODO - consider adding cuma as tunable, this 
    # means we won't be able to leak it here, 
    # and therefore we won't need to separately set it to the original value each time we call this
    # TODO - square how to use this with caching (but first benchmark if it's worth it)
    # HOW TO SELECT u0: p_work = replace(Initials(), prob.p, right_types), this might avoid the manual rebuilding

    # switch to the new drawn parameters
    uncertain_setter!(p_work, draws)

    # Solve the ODE
    try
        # warm up
        cuma_setter!(p_work, (2e-6, ))
        # FIXME - the dtmin is hardcoded here
        warm = solve(prob, Rosenbrock23(), p=p_work, dtmin=1e-12)
        # display(Plots.plot(warm))

        # TODO - caching
        # update u0 in p_work
        warm_u0 = warm.u[end]
        P = typeof(p_work).name.wrapper

        pvec = getfield(p_work, 1)
        u0_old   = getfield(p_work, 2)
        f3       = getfield(p_work, 3)
        f4       = getfield(p_work, 4)
        f5       = getfield(p_work, 5)
        f6       = getfield(p_work, 6)

        u0_setter! = setu(ns, unknowns(ns))

        # set u0
        # TODO - cache the u0 types
        T = eltype(warm_u0)
        # note we never modify u0_old, therefore u0 is never leaking into the next iteration 
        # and the warm up on the next call still uses the correct u0
        u0_work = similar(u0_old, T)
        copyto!(u0_work, u0_old)
        p_work = P(pvec, u0_work, f3, f4, f5, f6)
        u0_setter!(p_work[2], warm_u0)
        
        # faster?
        fast_prob = remake(prob, tspan=(0.0, 5.45))
        
        # @show p_work
        cuma_setter!(p_work, (2e-5, ))
        sol1 = solve(fast_prob, Rosenbrock23(), p=p_work; dtmin=1e-12, saveat=saveat)
        # display(Plots.plot(sol1))

        cuma_setter!(p_work, (0.0001, ))
        sol2 = solve(fast_prob, Rosenbrock23(), p=p_work; dtmin=1e-12, saveat=saveat)

        cuma_setter!(p_work, (0.001, ))
        sol3 = solve(fast_prob, Rosenbrock23(), p=p_work; dtmin=1e-12, saveat=saveat)
        # display(Plots.plot(sol3))
        
        # fixme - use 
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

# Initilize parameters using results from the RPA paper
# init_params_arr = [3.0,83.4743, 1.28e-8, 2.34, 2.75e3, 11.9586, 391.1627, 36.4063, 1.3, 3.9e-4, 8.7519e6, 0.51, 3.2, 7.1347, 89.0635, 7.0188, 17.7437, 4006.9, 0.6644]

init_params_draws = Dict(
    :σ => 3.0,
    :draws => [guess_map[p.name] for p in tunable_params],
)

Random.seed!(4)
nuts = NUTS(0.65,init_ϵ = 0.001)
chain_1 = sample(model2, nuts , MCMCSerial(), 3, 1, init_params = init_params_draws)

# rename the chain 
rename_map = Dict(
    Symbol("draws[$i]") => tunable_params[i].name
    for i in eachindex(tunable_params)
)
chain_named = replacenames(chain_1, rename_map)

f = open(string(@__DIR__)*"/minmtk_c30-reverseDiffAdBackend_jac0.jls", "w")
# f = open(string(@__DIR__)*"/minmtk_c31-mtkf.jls", "w")

serialize(f, chain_named)
close(f)

