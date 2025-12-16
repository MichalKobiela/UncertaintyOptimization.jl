using Test
using ModelingToolkit
using ModelingToolkit: t_nounits as t
using Distributions
using OrdinaryDiffEq
include("helpers/mock_rpa.jl")
using .MockRPA

@testset "Bayesian Inference Tests" begin
    
    @testset "Test simulation" begin
        
        model_def = MockRPA.mock_rpa_model()
        @mtkcompile sys = System(model_def.equations, t)
        model = UncertaintyOptimization.Model(model_def, sys)

        # Define simulation parameters
        init_cond = [1.0, 1.0]
                    
        # Ground truth values
        params = Dict(
                :beta_RA => 0.1,
                :beta_BA => 0.001, 
                :beta_AB => 0.01,
                :beta_BB => 0.001,    
            )
                    
        tspan = (0.0, 2.0)  # Short window for testing
                    
        # Run simulation
        sol = simulate!(model, init_cond, params, tspan)

        # Generate noisy observations
        t_obs = collect(range(1, stop = 2, length = 30))  # Change to range(1, 90, 30) for full validation
        randomized = VectorOfArray([sol(t_obs[i])[1] + 1*randn() for i in eachindex(t_obs)])
        data = convert(Array, randomized)

        setup_simulation!(model,
                            t_obs,
                            1,
                            init_cond,
                            params,
                            tspan;
                            solver=Euler(),
                            dt=0.01)


            println(model.prob.u0)       # initial conditions
            println(model.prob.tspan)    # time span
            println(model.prob.p)        # parameters

            println(equations(model.sys))
            println(unknowns(model.sys))
            println(parameters(model.sys))


        #Check the parameters have the correct starting values
        correct_values = [1.0, 1.0, 1.0, 1.0, 0.001, 1.0, 1.0, 1.0, 100.0, 
                          1.0, 0.01, 0.1, 100.0, 1.0, 0.001, 1.0, 1.0]

        @test all(model.prob.p .== correct_values)

   

    uncertain_syms = model.uncertain_params
    println(uncertain_syms)

    buffer_fcn = model.buffer_func

    # Order in sys
    #[beta_BA,  beta_AB, beta_RA, beta_BB]
    # [0.003, 0.004, 0.01, 0.02]

    #Order of uncertain_syms
    #[beta_RA, beta_BA, beta_BB, beta_AB]

    p_test = [0.01, 0.003, 0.02, 0.004] 
    buffer = model.buffer_func(p_test)
    println(parameters(model.sys))
    println(buffer)

    #TO DO: WRITE AUTO TEST TO MAKE SURE THAT THE PARAMETERS GET UPDATED BY NAME FROM THE BUFFER

 end

end

# @testset "InferenceProblem Constructor Tests" begin
    
#     # Setup - create a model
#     filename = joinpath(@__DIR__, "test-data", "test_RPA.yml")
#     model_def = UncertaintyOptimization.load_model_from_yaml(filename)
#     @mtkcompile sys = System(model_def.equations, t)
#     model =  UncertaintyOptimization.Model(model_def, sys)
    
#     @testset "Create InferenceProblem with defaults" begin
#         # Create inference problem
#         inf_prob =  UncertaintyOptimization.InferenceProblem(model)
        
#         @test typeof(inf_prob) == InferenceProblem
   
#         @test inf_prob.model === model
  
#         @test inf_prob.state_idx == 1  # Default: observe first state
#         @test typeof(inf_prob.noise_prior) <: InverseGamma  # Default noise prior
        
#         # Check that data fields are initially nothing
#         @test inf_prob.data === nothing
#         @test inf_prob.t_obs === nothing
#         @test inf_prob.chain === nothing
        
#         println("✅ InferenceProblem created with defaults")
#     end

# end

# @testset "InferenceProblem Data Setting Tests" begin
    
#     # Setup
#     filename = joinpath(@__DIR__, "test-data", "test_RPA.yml")
#     model_def = UncertaintyOptimization.load_model_from_yaml(filename)
#     @mtkcompile sys = System(model_def.equations, t)
#     model =  UncertaintyOptimization.Model(model_def, sys)
#     inf_prob =  UncertaintyOptimization.InferenceProblem(model)
    
#     @testset "set_data!() basic functionality" begin
#         data = [1.0, 2.0, 3.0, 4.0, 5.0]
#         times = [0.0, 1.0, 2.0, 3.0, 4.0]
        
#         set_data!(inf_prob, data, times)
        
#         println("✅ set_data!() stores data correctly")
#     end
    

#     @testset "set_data!() validation" begin
#         # Mismatched lengths should error
#         @test_throws ErrorException set_data!(inf_prob, [1.0, 2.0], [0.0, 1.0, 2.0])
        
#         # Empty data should error
#         @test_throws ErrorException set_data!(inf_prob, Float64[], Float64[])

#         # NaN in data should error
#         @test_throws ErrorException set_data!(inf_prob, [1.0, NaN, 3.0], [0.0, 1.0, 2.0])
        
#         # Inf in data should error
#         @test_throws ErrorException set_data!(inf_prob, [1.0, Inf, 3.0], [0.0, 1.0, 2.0])
        
#         println("set_data!() validates inputs")
#     end

# end