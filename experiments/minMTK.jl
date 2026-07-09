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
using SymbolicIndexingInterface: setp, variable_index
using Random
using Serialization
using CSV, Tables, DataFrames
using Plots
using Distributions
using AdvancedHMC: DenseEuclideanMetric
using SciMLBase: successful_retcode


Random.seed!(0);


const SOLVER = AutoTsit5(Rosenbrock23(autodiff=false))
const DTMIN = 1e-9

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

hill_eps = 1e-12
production_scale = 0.02

smooth_pos(x, eps) = 0.5 * (x + sqrt(x^2 + eps^2))

A_pos = smooth_pos(A, hill_eps)
B_pos = smooth_pos(B, hill_eps)

log_hill(numerator, denominator, n, eps) =
    1 / (1 + exp(n * (log(numerator + eps) - log(denominator + eps))))

hill_cuma = log_hill(kcymRtot * kx1, kx1 + cuma, nx1, hill_eps)
hill_B_kx2 = log_hill(kx2, B_pos, nx2, hill_eps)
hill_B_kx3 = log_hill(kx3, B_pos, nx2, hill_eps)
hill_A_kr = log_hill(A_pos, kr, nr, hill_eps)

A_cuma_factor = alpha_1 * hill_cuma + beta_1
A_B_factor = alpha_2 * hill_B_kx2 + beta_2
B_B_factor = alpha_3 * hill_B_kx3 + beta_3
B_A_factor = alpha_4 * hill_A_kr + beta_4

A_production = production_scale * A_cuma_factor * A_B_factor
B_production = production_scale * B_B_factor * B_A_factor
A_decay = production_scale * r1 * A_pos
B_decay = 0.1 * r2 * B_pos

eqs = [
    D(A) ~ A_production - A_decay,
    D(B) ~ B_production - B_decay,
]

@mtkcompile sys = System(eqs, t)

ordered_params = [p for p in parameters(sys)]

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
prob = ODEProblem(sys, merge(Dict(u0), initial_params), (0.0, 10.0), jac=true, simplify=false)

# prepare cuma setter and priors
tunable_params = [p for p in ordered_params if p in Set(ModelingToolkit.tunable_parameters(sys))]
tunable_priors = arraydist([prior_map[p.name] for p in tunable_params])

cuma_setter! = setp(sys, [getproperty(sys, :cuma),])
const INITIAL_STATE_INDICES = [variable_index(sys, state) for state in (sys.A, sys.B)]

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
        warm = solve(prob, SOLVER, p=p_work, dtmin=DTMIN, save_end=true, save_everystep=false, dense=false)

        warm_initials = warm.u[end][INITIAL_STATE_INDICES]
        p_work = replace(Initials(), p_work, warm_initials)
        production_prob = remake(prob; tspan=(first(saveat), last(saveat)))
        
        cuma_setter!(p_work, (2e-5, ))
        sol1 = solve(production_prob, SOLVER, p=p_work; dtmin=DTMIN, saveat=saveat)

        cuma_setter!(p_work, (0.0001, ))
        sol2 = solve(production_prob, SOLVER, p=p_work; dtmin=DTMIN, saveat=saveat)

        cuma_setter!(p_work, (0.001, ))
        sol3 = solve(production_prob, SOLVER, p=p_work; dtmin=DTMIN, saveat=saveat)

        if any(sol -> !successful_retcode(sol), (sol1, sol2, sol3))
            Turing.@addlogprob! -1e10
            return nothing
        end
        
        data ~ MvNormal(vcat(sol1[sys.A, :], sol2[sys.A, :], sol3[sys.A, :]), σ^2 * I)
    catch e
        # print(e)
        Turing.@addlogprob! -1e10
    end

    return nothing
end

# prepare data (time point and measurements)
data_frame = CSV.read(
    joinpath(@__DIR__, "reference", "RPA_real_data.csv"), DataFrame; normalizenames=true, stripwhitespace=true,
)
# select specific experimental data conditions
data_subset = vcat(
    data_frame.experession20,
    data_frame.experession100,
    data_frame.expression1000,
) .- 17.6 # adjust for background fluorescence


model = fit(data_subset, prob, data_frame.time, tunable_priors, InverseGamma(2, 3))

initial_params_draws = (;
    σ = 3.0,
    draws = [guess_map[p.name] for p in tunable_params],
)

Random.seed!(4)
sampler = NUTS(0.5,init_ϵ = 0.005, metricT = DenseEuclideanMetric)
chain_1 = sample(model, sampler , MCMCSerial(), 3000, 1, initial_params = [InitFromParams(initial_params_draws)])

rename_map = Dict(
    Symbol("draws[$i]") => tunable_params[i].name
    for i in eachindex(tunable_params)
)
chain_named = replacenames(chain_1, rename_map)

f = open(string(@__DIR__)*"/minmtk_r14_rewritten_order.jls", "w")
serialize(f, chain_named)
close(f)
