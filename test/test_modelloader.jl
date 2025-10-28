using Test
using YAML
using ModelingToolkit
using IOCapture
include("../src/ModelLoader.jl")

@testset "YAML Loading" begin
    # Test for missing file and an error being gracefully handled
    missing_file = "i-dont-exists-file.yml"
    output = IOCapture.capture() do 
        config = load_YAML(missing_file)
        @test config == nothing
    end
    println(output)
    @test occursin("❌  File with the name $missing_file not found, please check if the input path is correct and the file exists",output.output)
    
    # Test that it loads an a real file returning a Dict
    filename = joinpath(@__DIR__, "test-data", "test_RPA.yml")
    config = load_YAML(filename)
    @test config isa Dict

end

# Tests the build symbolics function to make sure that they are being set correctly
@testset "Build Symbolics" begin
    
    config = Dict(
        "parameters" => Dict(
            "k1" => Dict("role"=>"fixed","value"=>0.1),
            "k2" => Dict("role"=>"fixed","value"=>0.2),
            "k3" => Dict("role"=>"design","bounds"=>[0.0,1.0]),
            "k4" => Dict("role"=>"design","bounds"=>[0.0,1.0])
        ),
        "model" => Dict("states" => ["A", "B"]),
        "inputs" => Dict(
            "type" => "step",
            "t_threshold" => 5.0,
            "values" => [0.0, 1.0]
        )
    )

    symbolics = build_symbolics(config)

    # Check the structure 
    @test all(k -> k in (:states, :parameters, :input), keys(symbolics ))
    
    # Check that it returns all symbolics
    @test all(v -> v isa ParameterSpec, values(symbolics.parameters))
    @test all(v -> v.symbol isa Symbol, values(symbolics.parameters))
    @test all(v -> v isa Num, values(symbolics.states))
    @test symbolics[:input] isa Num
    @test occursin("ifelse", string(symbolics[:input]))


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
            "Y" => "beta*X - gamma*Y*(inputs)"
        )
    )

    syms = build_symbolics(config)
    eqs = build_equations(config, syms.states, syms.parameters, syms.input)
    
    # Should return a vector of equations 
    @test eqs isa Vector
    @test all(x -> x isa Equation, eqs)

end

