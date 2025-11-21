using Test
using ModelingToolkit
using ModelingToolkit: t_nounits as t

@testset "Test Model constructor" begin

    config = Dict(  
        "parameters" => Dict(
            "alpha" => Dict("role"=>"fixed","value"=>1.0),
            "beta"  => Dict("role"=>"uncertain","value"=>0.0),
            "gamma" => Dict("role"=>"design","value"=>0.1)
        ),
        "model" => Dict("states" => ["X", "Y"]),
        "inputs" => Dict("type" => "step", "t_threshold"=>5.0, "values"=>[0.0,1.0]),
        "equations" => Dict(
            "X" => "alpha*X + beta*Y - gamma*X",
            "Y" => "beta*X - gamma*Y*input"
        )
    )

    info = UncertaintyOptimization.get_model_info(config)
    syms = UncertaintyOptimization.build_symbolics(config)
    eqs = UncertaintyOptimization.build_equations(config, syms)

    model_def = UncertaintyOptimization.ModelDefinition(info.model_name,
                           info.model_description,
                           info.model_type,
                           eqs,
                           syms.states,
                           syms.parameters,
                           syms.input)

    @test typeof(model_def) == ModelDefinition
    @mtkcompile sys = System(model_def.equations, t)
        
        # Create the Model object
        model = UncertaintyOptimization.Model(model_def, sys)
        
        # Test that Model was created with correct fields
        @test typeof(model) == Model
        @test model.model_def == model_def
        @test model.sys == sys
        
        # Test that prob and sol are initially nothing
        @test model.prob === nothing
        @test model.sol === nothing
        
    
end

