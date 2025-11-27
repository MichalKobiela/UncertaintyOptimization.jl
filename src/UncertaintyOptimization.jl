module UncertaintyOptimization

include("ModelLoader.jl")
include("Model.jl")
include("AbstractInference.jl")
include("Inference.jl")

export load_model_from_yaml, ModelDefinition
export simulate!, Model, setup_simulation!
export InferenceProblem, set_data!

end
