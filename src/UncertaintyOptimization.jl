module UncertaintyOptimization

include("ModelLoader.jl")
include("Model.jl")
include("AbstractInference.jl")
include("Inference.jl")
include("algorithms/bayesian_inference.jl")

export load_model_from_yaml, ModelDefinition
export simulate!, Model, setup_simulation!
export InferenceProblem, set_data!
export BayesianSpec, run_inference

end
