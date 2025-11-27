using Test
using ModelingToolkit
using ModelingToolkit: t_nounits as t
include("helpers/mock_rpa.jl")
using .MockRPA

@testset "Test Model simulations" begin

    @testset "Test one off simulation" begin
    
        model_def = MockRPA.mock_rpa_model()

        @mtkcompile sys = System(model_def.equations, t)

        model = UncertaintyOptimization.Model(model_def, sys)

        u0 = [1.0, 1.0]

        params = Dict(
            :beta_RA => 0.1,
            :beta_AB => 0.001,
            :beta_BA => 0.01,
            :beta_BB => 0.001
        )
            
        tspan = (0.0, 100.0) 
            
        # Run simulation
        sol = simulate!(model, u0, params, tspan)
            
        # Check that solution exists
        @test sol !== nothing

    end

    @testset "Test setup simulation" begin
    
        model_def = MockRPA.mock_rpa_model()

        @mtkcompile sys = System(model_def.equations, t)

        model = UncertaintyOptimization.Model(model_def, sys)

        t_obs = collect(range(0.0, 100.0, length=10))
        params = Dict(
            :beta_RA => 0.1,
            :beta_AB => 0.001,
            :beta_BA => 0.01,
            :beta_BB => 0.001
        )
        
        UncertaintyOptimization.setup_simulation!(
            model, t_obs, 1, [1.0, 1.0], params, (0.0, 100.0)
        )
        
        # Call evaluate with uncertain params
        # Order: beta_RA, beta_AB, beta_BA, beta_BB
        predicted = UncertaintyOptimization.run_simulation(model, [0.1, 0.1, 0.1, 0.1])
        
        @test length(predicted) == length(t_obs)
        @test all(isfinite, predicted)
        

    end
    
end

# @testset "Test Model constructor" begin

#     config = Dict(  
#         "parameters" => Dict(
#             "alpha" => Dict("role"=>"fixed","value"=>1.0),
#             "beta"  => Dict("role"=>"uncertain","value"=>0.0),
#             "gamma" => Dict("role"=>"design","value"=>0.1)
#         ),
#         "model" => Dict("states" => ["X", "Y"]),
#         "inputs" => Dict("type" => "step", "t_threshold"=>5.0, "values"=>[0.0,1.0]),
#         "equations" => Dict(
#             "X" => "alpha*X + beta*Y - gamma*X",
#             "Y" => "beta*X - gamma*Y*input"
#         )
#     )

#     info = UncertaintyOptimization.get_model_info(config)
#     syms = UncertaintyOptimization.build_symbolics(config)
#     eqs = UncertaintyOptimization.build_equations(config, syms)

#     model_def = UncertaintyOptimization.ModelDefinition(info.model_name,
#                            info.model_description,
#                            info.model_type,
#                            eqs,
#                            syms.states,
#                            syms.parameters,
#                            syms.input)

#     @test typeof(model_def) == ModelDefinition
#     @mtkcompile sys = System(model_def.equations, t)
        
#     # Create the Model object
#     model = UncertaintyOptimization.Model(model_def, sys)
        
#     # Test that Model was created with correct fields
#     @test typeof(model) == Model
#     @test model.model_def == model_def
#     @test model.sys == sys
        
#     # Test that prob and sol are initially nothing
#     @test model.prob === nothing
#     @test model.sol === nothing
        
    
# end

# @testset "Model simulate! Tests" begin
    
#     # Setup - create a model for all tests
#     filename = joinpath(@__DIR__, "test-data", "test_RPA.yml")
#     model_def = UncertaintyOptimization.load_model_from_yaml(filename)
#     @mtkcompile sys1 = System(model_def.equations, t)
#     model = UncertaintyOptimization.Model(model_def, sys1)

#     # Define simulation parameters
#     init_cond = [1.0, 1.0]  # Initial conditions for [R, A]
        
#     # Parameters to simulate with (ground truth values)
#     params = Dict(
#         :beta_RA => 0.1,
#         :beta_AB => 0.001,
#         :beta_BA => 0.01,
#         :beta_BB => 0.001
#     )
        
#     tspan = (0.0, 100.0)  # Simulate from t=0 to t=10
        
#     # Run simulation
#     sol = simulate!(model, init_cond, params, tspan)
        
#     # Check that solution exists
#     @test sol !== nothing
        
#     # Check that model.sol was updated
#     @test model.sol === sol
        
#     # Check that model.prob was created
#     @test model.prob !== nothing
        


# end