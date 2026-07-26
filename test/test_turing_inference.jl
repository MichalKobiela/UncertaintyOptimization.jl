using Test
using ModelingToolkit
using ModelingToolkit: t_nounits as t
using OrdinaryDiffEq
using Distributions
using Turing
using DataFrames
using Random

@testset "Turing inference" begin
    @testset "conditions on observed data" begin
        @variables X(t)
        k = UncertaintyOptimization.create_param("k"; tunable=true)

        eqs = [
            Differential(t)(X) ~ k
        ]

        params = Dict{Symbol, UncertaintyOptimization.ParameterSpec}(
            :k => UncertaintyOptimization.ParameterSpec(
                "k",
                k,
                :uncertain,
                0.0,
                nothing,
                nothing,
                Dict("distribution" => "uniform", "lower" => 0.0, "upper" => 2.0),
                nothing,
            ),
        )

        model_def = UncertaintyOptimization.ModelDefinition(
            "LinearInference",
            "Tiny model for testing Turing data conditioning",
            :ODE,
            eqs,
            Dict(:X => X),
            params,
            nothing,
        )

        @mtkcompile linear_inference_sys = System(model_def.equations, t)
        model = UncertaintyOptimization.Model(model_def, linear_inference_sys)

        sim_spec = SimulationSpec(
            t_obs = [0.5, 1.0],
            obs_state = :X,
            initial_conditions = (0.0,),
            tspan = (0.0, 1.0),
            uncertain_param_values = Dict(:k => 1.0),
            solver = Euler(),
            solver_opts = (dt = 0.05,),
        )

        spec = TuringSpec(
            simulation = sim_spec,
            data = [0.25, 0.9],
            noise_prior = InverseGamma(2, 3),
            noise_initial = 1.0,
            sampler = NUTS(0.65, init_ϵ = 0.01),
            n_samples = 1,
            n_chains = 1,
            sampling_method = MCMCSerial(),
        )

        Random.seed!(1)
        chain = run_inference(model, spec)
        chain_df = DataFrame(chain)

        @test hasproperty(chain_df, :loglikelihood)
        @test all(isfinite, chain_df.loglikelihood)
        @test !all(iszero, chain_df.loglikelihood)
    end
end
