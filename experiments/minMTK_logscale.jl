using Revise
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEq
using CSV, Tables
using Turing
using SciMLBase: successful_retcode
using SciMLStructures: Tunable, canonicalize, replace
using SymbolicIndexingInterface
using Random
using Serialization
using Plots
using DataFrames
using Distributions

const SOLVER = Rosenbrock23()

Random.seed!(0)

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
    D(B) ~ 0.02 * (alpha_3 * (max(B, 0)^nx2) / (kx3^nx2 + max(B, 0)^nx2) + beta_3) *
          (alpha_4 / (1 + (max(A, 0) / kr)^nr) + beta_4) -
          0.1 * r2 * max(B, 0),
    D(A) ~ 0.02 * (alpha_1 / (1 + (kcymRtot / (1 + cuma / kx1))^nx1) + beta_1) *
          (alpha_2 * (max(B, 0)^nx2) / (kx2^nx2 + max(B, 0)^nx2) + beta_2) -
          0.02 * r1 * max(A, 0),
]

# Keep MTK as lightweight as possible here: no @mtkcompile, no structural_simplify.
@mtkcompile ns = System(eqs, t)
# ns = complete(ns)
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
    :cuma => 2e-6,
)

# Parameters that are strictly positive and span many orders of magnitude are sampled on log scale.
const LOG_SCALE_PARAMS = Set([
    :alpha_1,
    :alpha_2,
    :alpha_3,
    :alpha_4,
    :beta_1,
    :beta_2,
    :beta_3,
    :beta_4,
    :kx1,
    :kx2,
    :kr,
    :r1,
    :r2,
])

# Positive lower bounds for the log-space priors.
# These are chosen to keep the support broad while avoiding log(0).
log_bound_map = Dict{Symbol,Tuple{Float64,Float64}}(
    :alpha_1 => (1e-3, 2000.0),
    :alpha_2 => (1e-3, 250.0),
    :alpha_3 => (1e-3, 1e4),
    :alpha_4 => (1e-3, 1e13),
    :beta_1  => (1e-6, 200.0),
    :beta_2  => (1e-8, 100.0),
    :beta_3  => (1e-6, 5e3),
    :beta_4  => (1e-6, 5000.0),
    :kx1     => (1e-12, 3e-8),
    :kx2     => (1e-3, 1e4),
    :kr      => (1e-4, 100.0),
    :r1      => (1e-3, 1000.0),
    :r2      => (1e-3, 1000.0),
)

linear_prior_map = Dict{Symbol,Distribution}(
    :nx1 => Uniform(1.0, 5.0),
    :nx2 => Uniform(1.0, 10.0),
    :nr  => Uniform(1.0, 100.0),
)

latent_name(sym::Symbol) = sym in LOG_SCALE_PARAMS ? Symbol("log_", sym) : sym
latent_value(sym::Symbol, x) = sym in LOG_SCALE_PARAMS ? log(x) : x
actual_value(sym::Symbol, x) = sym in LOG_SCALE_PARAMS ? exp(x) : x

function latent_prior(sym::Symbol)
    if sym in LOG_SCALE_PARAMS
        lo, hi = log_bound_map[sym]
        return Uniform(log(lo), log(hi))
    end
    return linear_prior_map[sym]
end

u0 = [A => 24.0, B => 350.0]
init_params = Dict([p => guess_map[p.name] for p in ordered_params])

prob = ODEProblem(ns, merge(Dict(u0), init_params), (0.0, 10.0), jac=true, simplify=false)

tunable_set = Set(ModelingToolkit.tunable_parameters(ns))
tunable_params = [p for p in parameters(ns) if p in tunable_set]

multiparams_Nums = Vector{Num}(undef, length(tunable_params))
for (i, param) in enumerate(tunable_params)
    multiparams_Nums[i] = getproperty(ns, param.name)
end

uncertain_setter! = setp(ns, multiparams_Nums)
cuma_setter! = setp(ns, [getproperty(ns, :cuma)])
tunable_ps, _, _ = canonicalize(Tunable(), prob.p)

latent_priors = arraydist([latent_prior(p.name) for p in tunable_params])

state_order = unknowns(ns)
A_idx = findfirst(isequal(A), state_order)

function to_actual_draws(raw_draws, tunable_params)
    T = eltype(raw_draws)
    draws = Vector{T}(undef, length(raw_draws))
    for i in eachindex(raw_draws)
        draws[i] = actual_value(tunable_params[i].name, raw_draws[i])
    end
    return draws
end

@model function fit(data::AbstractVector, prob, saveat::AbstractVector, distributions, ig;
    force_values=nothing)

    σ ~ ig
    raw_draws ~ distributions

    if !isnothing(force_values)
        raw_draws = force_values
    end

    draws = to_actual_draws(raw_draws, tunable_params)
    T = eltype(draws)

    right_types = T.(tunable_ps)
    p_work = replace(Tunable(), prob.p, right_types)
    uncertain_setter!(p_work, draws)

    cuma_setter!(p_work, (2e-6,))
    warm = nothing
    try
        warm = solve(prob, SOLVER, p=p_work, dtmin=1e-12)
    catch e
        println(e)
        Turing.@addlogprob! -1e10
        return
    end 
    

    warm_u0 = warm.u[end]
    P = typeof(p_work).name.wrapper

    pvec = getfield(p_work, 1)
    u0_old = getfield(p_work, 2)
    f3 = getfield(p_work, 3)
    f4 = getfield(p_work, 4)
    f5 = getfield(p_work, 5)
    f6 = getfield(p_work, 6)

    u0_setter! = setu(ns, unknowns(ns))
    T_u0 = eltype(warm_u0)
    u0_work = similar(u0_old, T_u0)
    copyto!(u0_work, u0_old)
    p_work = P(pvec, u0_work, f3, f4, f5, f6)
    u0_setter!(p_work[2], warm_u0)

    cuma_setter!(p_work, (2e-5,))
    sol1 = solve(prob, SOLVER, p=p_work; dtmin=1e-12, saveat=saveat)

    cuma_setter!(p_work, (0.0001,))
    sol2 = solve(prob, SOLVER, p=p_work; dtmin=1e-12, saveat=saveat)

    cuma_setter!(p_work, (0.001,))
    sol3 = solve(prob, SOLVER, p=p_work; dtmin=1e-12, saveat=saveat)

    # all solves succeeded
    if any(sol -> !successful_retcode(sol), [sol1, sol2, sol3])
        Turing.@addlogprob! -1e10
        return
    end

    predicted = vcat(sol1[A_idx, :], sol2[A_idx, :], sol3[A_idx, :])

    # predicted/data are vectors of same length
    if !(predicted isa AbstractVector)
        println("predicted is not abstract")
        Turing.@addlogprob! -1e10
        return
    end

    if length(predicted) != length(data)
        println("different lengths")
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

time = CSV.read(string(@__DIR__) * "/RPA_real_data/time_points.csv", DataFrame)[!, 1]
data = Matrix(CSV.read(string(@__DIR__) * "/RPA_real_data/data.csv", DataFrame))
background_fluorescence = 17.6
data = data .- background_fluorescence
data_subset = vcat(data[:, 2], data[:, 5], data[:, 9])

model2 = fit(data_subset, prob, time, latent_priors, InverseGamma(2, 3))

init_params_draws = Dict(
    :σ => 3.0,
    :raw_draws => [latent_value(p.name, guess_map[p.name]) for p in tunable_params],
)

Random.seed!(4)
nuts = NUTS(0.55, init_ϵ=0.002)
chain_1 = sample(model2, nuts, MCMCSerial(), 3000, 1, init_params=init_params_draws)

rename_map = Dict(
    Symbol("raw_draws[$i]") => latent_name(tunable_params[i].name)
    for i in eachindex(tunable_params)
)
chain_named = replacenames(chain_1, rename_map)

println("Sampled on log scale: ", sort!(collect(LOG_SCALE_PARAMS)))
println("Latent variable names in saved chain are prefixed with log_ where applicable.")

f = open(string(@__DIR__) * "/minmtk_logscale_c2.jls", "w")
serialize(f, chain_named)
close(f)
