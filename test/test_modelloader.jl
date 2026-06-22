using Test
using YAML
using ModelingToolkit



@testset "YAML Loading" begin
    # Test for missing file and an error being gracefully handled
    missing_file = "i-dont-exists-file.yml"
    @test_logs (:warn, r"File not found") begin
        config = UncertaintyOptimization.load_YAML(missing_file)
        @test config == nothing
    end
    
    # Test that it loads an a real file returning a Dict
    filename = joinpath(@__DIR__, "test-data", "test_RPA.yml")
    if isfile(filename)
        config = UncertaintyOptimization.load_YAML(filename)
        @test config isa Dict
    else
        @info "Skipping test: $filename not found (likely CI environment)"
    end

end

@testset "Symbolics conversion" begin

    param = UncertaintyOptimization.create_param("k")
    @parameters k

    @test isequal(param,k)

    var = UncertaintyOptimization.create_param("A")
    @variables A

    @test isequal(var,A)

    
end

# Tests the build symbolics function to make sure that they are being set correctly
@testset "Build Symbolics" begin


    config = Dict(
        "parameters" => Dict(
            "k1" => Dict("role"=>"fixed","value"=>0.1),
            "k2" => Dict("role"=>"fixed","value"=>0.2),
            "k3" => Dict("role"=>"design","bounds"=>[0.0,1.0]),
            "k4" => Dict("role"=>"design","bounds"=>[0.0,1.0]),
            "k5" => Dict(
                "role" => "fixed",
                "value" => 0.5,
                "design" => Dict(
                    "value" => 0.6,
                ),
            ),
            "kx2" => Dict(
                "role" => "uncertain",
                "value" => 36.4063,
                "prior" => Dict(
                    "distribution" => "uniform",
                    "lower" => 0.0,
                    "upper" => 1e4,
                ),
                "design_optimise" => Dict(
                    "scalers" => [1, 2, 3, 4],
                ),
            ),
            "kx3" => Dict(
                "role" => "fixed",
                "value" => 4006.9,
                "design_optimise" => Dict(
                    "scalers" => "0.1:0.03:0.16",
                ),
            ),
            "cuma" => Dict(
                "role" => "fixed",
                "warmup_value" => 2e-6,
                "value" => 0.0001,
                "design" => Dict(
                    "warmup_value" => 2e-6,
                    "value" => 0.0003,
                ),
            )
        ),
        "model" => Dict("states" => ["A", "B"]),
        "inputs" => Dict(
            "type" => "step",
            "t_threshold" => 5.0,
            "values" => [0.0, 1.0]
        )
    )

    symbolics = UncertaintyOptimization.build_symbolics(config)


    for pname in ["k1", "k2", "k3", "k4", "k5", "kx2", "kx3", "cuma"]
        param = Symbol(pname)
        @test isequal(symbolics.parameters[param].symbol, Symbolics.unwrap(first(@parameters $param)))
    end

    @test isnothing(symbolics.parameters[:k1].design)
    @test isnothing(symbolics.parameters[:k1].design_optimise)
    @test symbolics.parameters[:cuma].design isa UncertaintyOptimization.Design
    @test symbolics.parameters[:cuma].design.warmup_value == 2e-6
    @test symbolics.parameters[:cuma].design.value == 0.0003
    @test symbolics.parameters[:cuma].warmup_value == 2e-6
    @test symbolics.parameters[:cuma].value == 0.0001
    @test isnothing(symbolics.parameters[:k5].design.warmup_value)
    @test symbolics.parameters[:k5].design.value == 0.6
    @test UncertaintyOptimization.get_warmup_params(symbolics.parameters) == Dict(:cuma => 2e-6)
    @test UncertaintyOptimization.get_warmup_params(symbolics.parameters; design=true) == Dict(:cuma => 2e-6)
    @test symbolics.parameters[:kx2].design_optimise == (1.0, 2.0, 3.0, 4.0)
    @test symbolics.parameters[:kx3].design_optimise == (0.1, 0.13, 0.16)

    for (s_sym, var_obj) in symbolics.states
        @test typeof(var_obj) <: SymbolicUtils.BasicSymbolic
    end

end

 
@testset "Build Equations" begin
    config = Dict(  
        "parameters" => Dict(
            "alpha" => Dict("role"=>"fixed","value"=>1.0),
            "beta"  => Dict("role"=>"optimizable","value"=>0.0),
            "gamma" => Dict("role"=>"fixed","value"=>0.1)
        ),
        "model" => Dict("states" => ["X", "Y"]),
        "inputs" => Dict("type" => "step", "t_threshold"=>5.0, "values"=>[0.0,1.0]),
        "equations" => Dict(
            "X" => "alpha*X + beta*Y - gamma*X",
            "Y" => "beta*X - gamma*Y*input"
        )
    )

    syms = UncertaintyOptimization.build_symbolics(config)
    eqs = UncertaintyOptimization.build_equations(config, syms)
    
    @test all(e -> e isa ModelingToolkit.Equation, eqs)
    
    
end
