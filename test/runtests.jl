using UncertaintyOptimization
using Test

@testset "UncertaintyOptimization.jl" begin
    include("test_modelloader.jl")
    include("test_model.jl")
end
