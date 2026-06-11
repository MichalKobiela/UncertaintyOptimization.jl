module UncertaintyOptimization

include("ModelLoader.jl")
include("Model.jl")
include("AbstractSimulation.jl")
include("AbstractInference.jl")
include("AbstractScan.jl")
include("Inference.jl")
include("algorithms/turing_inference.jl")
include("algorithms/grid_scan.jl")

export load_model, ModelDefinition
export simulate!, Model, setup_simulation!
export InferenceProblem, set_data!
export SimulationSpec, TuringSpec, ThompsonGridSpec, CartesianSampler, run_inference, run_scan

end
