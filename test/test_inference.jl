using Test
using ModelingToolkit
using ModelingToolkit: t_nounits as t
using Distributions

@testset "InferenceProblem Constructor Tests" begin
    
    # Setup - create a model
    model_def = load_model_from_yaml("./test/test-data/test_RPA.yml")
    @mtkcompile sys = System(model_def.equations, t)
    model = Model(model_def, sys)
    
    @testset "Create InferenceProblem with defaults" begin
        # Create inference problem
        inf_prob = InferenceProblem(model)
        
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