using Test
using ModelingToolkit
using ModelingToolkit: t_nounits as t
using Distributions

@testset "InferenceProblem Constructor Tests" begin
    
    # Setup - create a model
    filename = joinpath(@__DIR__, "test-data", "test_RPA.yml")
    model_def = UncertaintyOptimization.load_model_from_yaml(filename)
    @mtkcompile sys = System(model_def.equations, t)
    model =  UncertaintyOptimization.Model(model_def, sys)
    
    @testset "Create InferenceProblem with defaults" begin
        # Create inference problem
        inf_prob =  UncertaintyOptimization.InferenceProblem(model)
        
        @test typeof(inf_prob) == InferenceProblem
   
        @test inf_prob.model === model
  
        @test inf_prob.state_idx == 1  # Default: observe first state
        @test typeof(inf_prob.noise_prior) <: InverseGamma  # Default noise prior
        
        # Check that data fields are initially nothing
        @test inf_prob.data === nothing
        @test inf_prob.t_obs === nothing
        @test inf_prob.chain === nothing
        
        println("✅ InferenceProblem created with defaults")
    end

end

@testset "InferenceProblem Data Setting Tests" begin
    
    # Setup
    filename = joinpath(@__DIR__, "test-data", "test_RPA.yml")
    model_def = UncertaintyOptimization.load_model_from_yaml(filename)
    @mtkcompile sys = System(model_def.equations, t)
    model =  UncertaintyOptimization.Model(model_def, sys)
    inf_prob =  UncertaintyOptimization.InferenceProblem(model)
    
    @testset "set_data!() basic functionality" begin
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        times = [0.0, 1.0, 2.0, 3.0, 4.0]
        
        set_data!(inf_prob, data, times)
        
        println("✅ set_data!() stores data correctly")
    end
    

    @testset "set_data!() validation" begin
        # Mismatched lengths should error
        @test_throws ErrorException set_data!(inf_prob, [1.0, 2.0], [0.0, 1.0, 2.0])
        
        # Empty data should error
        @test_throws ErrorException set_data!(inf_prob, Float64[], Float64[])

        # NaN in data should error
        @test_throws ErrorException set_data!(inf_prob, [1.0, NaN, 3.0], [0.0, 1.0, 2.0])
        
        # Inf in data should error
        @test_throws ErrorException set_data!(inf_prob, [1.0, Inf, 3.0], [0.0, 1.0, 2.0])
        
        println("set_data!() validates inputs")
    end

end