using OrdinaryDiffEq
using CSV, Tables
using LinearAlgebra: I
using Turing
using Random
using Serialization
using Plots
using DataFrames



Random.seed!(0);
const SOLVER = Rosenbrock23()


tunable_params = [:alpha_1, :kx1, :nx1, :beta_1, :alpha_2, :kx2, :nx2, 
    :beta_2, :alpha_4, :kr, :nr, :beta_4, :r1, :r2, :alpha_3, :beta_3]
fixed_params = [:kx3, :kcymRtot, :cuma]

# the tunable parameters are listed first
order = [tunable_params ; fixed_params]

tunable_last_idx = length(tunable_params)
cuma_idx = findfirst(isequal(:cuma), order)


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


@model function fit(
    data::AbstractVector, 
    prob, 
    saveat::AbstractVector)

    σ ~ InverseGamma(2, 3)
    
    alpha_1 ~ truncated(Distributions.Uniform(0.0, 2000.0), lower = 0.0)
    kx1 ~ truncated(Distributions.Uniform(0, 3.0e-8), lower = 0.0)
    nx1 ~ truncated(Distributions.Uniform(1.0, 5.0), lower = 0.0)
    beta_1  ~ truncated(Distributions.Uniform(0, 200.0), lower = 0.0)
    alpha_2 ~ truncated(Distributions.Uniform(0.0, 250.0), lower = 0.0)
    kx2 ~ truncated(Distributions.Uniform(0.0, 10000), lower = 0.0)
    nx2 ~ truncated(Distributions.Uniform(1.0, 10.0), lower = 0.0)
    beta_2 ~ truncated(Distributions.Uniform(0, 100.0), lower = 0.0)
    alpha_4 ~ truncated(Distributions.Uniform(0, 1.0e13), lower = 0.0)
    kr ~ truncated(Distributions.Uniform(0.0, 100.0), lower = 0.0)
    nr ~ truncated(Distributions.Uniform(1.0, 100.0), lower = 0.0)
    beta_4 ~ truncated(Distributions.Uniform(0,5000), lower = 0.0)
    r1 ~ truncated(Distributions.Uniform(0.0,1000.0), lower = 0.0)
    r2 ~ truncated(Distributions.Uniform(0.0, 1000.0), lower = 0.0)
    alpha_3 ~ truncated(Distributions.Uniform(0.0, 10000.0), lower = 0.0)
    beta_3 ~ truncated(Distributions.Uniform(0.0, 5000.0), lower = 0.0)

    p_work = [alpha_1, kx1, nx1, beta_1, alpha_2, kx2, nx2, beta_2, alpha_4, kr, nr, beta_4, r1, r2, alpha_3, beta_3,
        # fixed [:kx3, :kcymRtot, :cuma]
                4006.9, 2.75e3,     2e-6
    ]

    try
        warm = solve(prob, SOLVER, p=p_work, u0=u0, dtmin=1e-12)
        warm_u0 = warm[end]

        p_work[cuma_idx] = 2e-5
        sol1 = solve(prob, SOLVER; p=p_work, u0=warm_u0, dtmin=1e-12, saveat=saveat)

        p_work[cuma_idx] = 0.0001
        sol2 = solve(prob, SOLVER; p=p_work, u0=warm_u0, dtmin=1e-12, saveat=saveat)

        p_work[cuma_idx] = 0.001
        sol3 = solve(prob, SOLVER; p=p_work, u0=warm_u0, dtmin=1e-12, saveat=saveat)
        
        data ~ MvNormal(vcat(sol1[A_idx,:], sol2[A_idx,:], sol3[A_idx,:]), σ^2 * I)
    catch e
        print(e)
        Turing.@addlogprob! -1e10
    end

    return nothing
end

data_frame = CSV.read(
    joinpath(@__DIR__, "RPA_real_data.csv"), DataFrame; normalizenames=true, stripwhitespace=true)
# select specific modelled data
data_subset = vcat(
    data_frame.experession20,
    data_frame.experession100,
    data_frame.expression1000,
) .- 17.6 # subtract background fluorescence

model = fit(data_subset, prob, data_frame.time)

initial_param_names = (:σ, tunable_params...)
initial_param_values = (3.0, (guess_map[param] for param in tunable_params)...)
initial_params = [InitFromParams(NamedTuple{initial_param_names}(initial_param_values))]

Random.seed!(4)
nuts = NUTS(0.65,init_ϵ = 0.001)
chain = sample(model, nuts , MCMCSerial(), 3000, 1, initial_params = initial_params)

f = open(string(@__DIR__)*"/reproduce_original_mk11_sanityJ12.jls", "w")
serialize(f, chain)
close(f)
