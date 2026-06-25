using YAML
using ModelingToolkit

const IV = ModelingToolkit.t_nounits


"""
Model loading utilities for turning a YAML experiment description into the
symbolic pieces needed by the rest of the package.

The loader is the first stage of the workflow: it reads the declarative model
definition, validates the required sections, creates ModelingToolkit states and
parameters, and returns a `ModelDefinition`. Simulation settings such as time
points, initial conditions, solver choice, and observed states are supplied
later through `SimulationSpec`.
"""

# -------------------------------------------------------------------------
# Struct Definitions
# -------------------------------------------------------------------------

"""
    Design

Design-stage values for a parameter.

`warmup_value` is used during the optional warmup solve in the design stage.
`value` is used during the production solve after warmup. These fields are read
from a parameter's `design:` block in YAML and allow a design/evaluation run to
use values that differ from the ordinary simulation or inference values.
"""
struct Design
    warmup_value:: Union{Nothing, Float64} 
    value:: Union{Nothing, Float64, Tuple{Vararg{Float64}}}   
end

# Currently immutable but we can make them mutable if required later
"""
    ParameterSpec

Parameter metadata parsed from YAML.

Each parameter has a `role`:

- `:fixed` parameters keep the scalar `value` supplied in YAML.
- `:uncertain` parameters become tunable ModelingToolkit parameters and must
  provide a prior when used for Turing inference.
- `:design` parameters are candidates for design-stage scans and may also
  define `bounds`, staged `value`s, `warmup_value`, or a nested `design` block.

Scalar values are used directly when building an `ODEProblem`. Tuple-valued
parameters represent staged production solves: `simulate!` runs one solve per
tuple entry after any warmup solve has completed.
"""
struct ParameterSpec
    name::String # paramater name
    # TODO - why is symbol Any? 
    symbol::Any # parameter symbolic
    role:: Symbol # whether :fixed, :uncertain, :design
    value:: Union{Nothing, Float64, Tuple{Vararg{Float64}}} # value of the paramater if provided
    warmup_value:: Union{Nothing, Float64, Tuple{Float64}} # value of the paramater if provided
    bounds::Union{Nothing, Tuple{Float64,Float64}} # bounds for the parameter if provided
    prior:: Union{Nothing, Dict}
    design:: Union{Nothing, Design}
    design_optimise:: Union{Nothing, Tuple{Vararg{Float64}}}
end

"""
    ModelDefinition

Symbolic model, states, parameters, and input expression parsed from YAML.

A `ModelDefinition` is intentionally declarative. It stores the parsed model
name and description, the model type, ModelingToolkit equations, symbolic
states, parameter specifications, and the generated input signal. Compile it
with ModelingToolkit and wrap it in `Model` before simulation, inference, or
scan stages:

```julia
model_def = load_model("model.yml")
@mtkcompile sys = System(model_def.equations, t)
model = Model(model_def, sys)
```
"""
struct ModelDefinition
    model_name::String
    model_description::String
    model_type::Symbol
    equations::Vector{Equation}
    states::Dict{Symbol, Any}
    parameters::Dict{Symbol, ParameterSpec}
    input::Any
end


# -------------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------------

"""
    load_YAML(filename::String) -> Dict

Load and parse a YAML file.

This is a low-level helper used by `load_model`. It returns the YAML content as
a dictionary when `filename` exists, and returns `nothing` after emitting a
warning when the file cannot be found.
"""

function load_YAML(filename:: String)
    if isfile(filename)
        return YAML.load_file(filename)
    else
        @warn "File not found; check that the input path is correct and the file exists" filename
        return nothing
    end
end

"""
    create_param(x)

Helper to convert a char read from the YAML to an @parameter required for the ModellingToolkit system

"""

function create_param(x; tunable::Bool=false)
    sym = Symbol(x)
    Symbolics.unwrap(first(@parameters $sym [tunable = tunable]))
end


"""

    create_var(name::String)

Create a ModelingToolkit variable symbol, optionally time-dependent.

"""
function create_var(x, iv::Num)
    sym = Symbol(x)
    Symbolics.unwrap(first(@variables $sym(iv)))
end


# -------------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------------

"""
    validate_YAML(config::Dict) -> Bool

Validate the sections that are required to build a symbolic model.

The current loader expects `experiment`, `model`, `parameters`, and `equations`
sections. It also checks that every equation targets a declared state and that
the equation strings parse as Julia expressions.
"""
function validate_YAML(config::Dict)
    # Check the required tags are there
    required_tags = ["experiment", "model", "parameters", "equations"]
    for tag in required_tags
        if !haskey(config, tag)
            @error "Missing required section in YAML" tag
            error(:"❌ Missing required section in YAML: '$tag'")
        end
    end

    # Check that the states in the equations match the states in the model and syntax is okay
    eqs = config["equations"]
    for (state, eq_str) in eqs
        if !(state in config["model"]["states"])
            error("❌ Equation in YAML references undefined state: $state")
        end
        try
            Meta.parse(eq_str)
        catch e
            error("❌ Invalid syntax in YAML equation for $state: $(e.msg)")
        end
    end
    
    @info "Valid YAML"
    return true

end

"""
    parse_values(x)

Convert YAML vectors to tuples and leave scalar values unchanged.
"""
function parse_values(x)
    if isnothing(x)
        return x
    elseif x isa AbstractVector
        return tuple(x...)
    else
        return x
    end
end

"""
    parse_design_optimise_values(x) -> Tuple

Parse design scan values from a vector or `start:step:stop` string.
"""
function parse_design_optimise_values(x)::Tuple{Vararg{Float64}}
    values = if x isa AbstractVector
        Float64.(x)
    elseif x isa AbstractString
        parts = split(x, ":")
        if length(parts) != 3
            error("design_optimise.scalers must use start:step:stop notation. Got: $x")
        end

        start, step, stop = parse.(Float64, strip.(parts))
        if step == 0.0
            error("design_optimise.values step must not be zero")
        end

        collect(start:step:stop)
    else
        error("design_optimise.scalers must be either a vector or a start:step:stop string. Got $(typeof(x)).")
    end

    if isempty(values) || !all(isfinite, values)
        error("design_optimise.scalers must contain finite values")
    end

    return tuple(values...)
end

# -------------------------------------------------------------------------
# Model Symbolic Construction
# -------------------------------------------------------------------------

"""
    build_symbolics(config)

Build ModelingToolkit variables, parameters, and input expression from YAML data.
"""
function build_symbolics(config::Dict) 

  #  # Symbolic states
    state_symbs = config["model"]["states"] # Read in states from YAML and convert to MTK variable
    state_map = Dict(Symbol(s) => create_var(s, IV) for s in state_symbs)

    # Get parameters specifications
    param_specs = Dict{Symbol, ParameterSpec}()

    # TODO - it would be nice to keep the same order as YAML
    for (pname_str, pinfo) in config["parameters"]   
        role = Symbol(pinfo["role"])

        tunable = role == :uncertain
        param = create_param(pname_str; tunable=tunable)   # create MTK parameter

        value = get(pinfo, "value", nothing)
        warmup_value = get(pinfo, "warmup_value", nothing)
        bounds = haskey(pinfo, "bounds") ? tuple(pinfo["bounds"]...) : nothing
        prior = get(pinfo, "prior", nothing)
        design = get(pinfo, "design", nothing)
        design_optimise = get(pinfo, "design_optimise", nothing)
        
        # either a float or a tuple
        value = parse_values(value) 
        warmup_value = parse_values(warmup_value)

        # add the design values
        if !isnothing(design)
            design = Design(parse_values(get(design, "warmup_value", nothing)), parse_values(design["value"]))
        end

        if !isnothing(design_optimise)
            design_optimise = parse_design_optimise_values(design_optimise["scalers"])
        end

        param_specs[Symbol(pname_str)] = ParameterSpec(pname_str, param, role, value, warmup_value, bounds, prior, design, design_optimise)
    end

    # Makes an input signal defined by the YAML
    if config["inputs"]["type"] == "step"
        input = ifelse(IV < config["inputs"]["t_threshold"],
                config["inputs"]["values"][1],
                config["inputs"]["values"][2])
    else
        error("❌ Unsupported input signal type: $(config["type"])")
    end

    return (states=state_map, parameters=param_specs, input=input)

end

# -------------------------------------------------------------------------
# Equation Construction
# -------------------------------------------------------------------------
"""
    expr_to_symbolic(expr_str, symbolics)

Evaluate a YAML equation expression against the model's symbolic environment.
"""
function expr_to_symbolic(expr_str::String, symbolics)
    # Build an sandbox mapping symbols -> symbolic variables
    # TODO - comment: not actually a sandbox if eval global const, 
    env = Dict{Symbol, Any}()

    for (k, v) in symbolics.states
        env[k] = v
    end

    for (k, v) in symbolics.parameters
        env[k] = v.symbol
    end

    if symbolics.input !== nothing
        env[:input] = symbolics.input
    end

    parsed = Meta.parse(expr_str)

    # Evaluate it symbolically - might need to watch out here
    return Base.invokelatest(eval, Expr(:block, [:($(k) = $(v)) for (k, v) in env]..., parsed))
end

"""
    expr_to_symbolic(expr, symbolics)

Convert an expression to the symbolic form used by model equations.
"""
function expr_to_symbolic(expr::Expr, symbolics)
    # convert Expr to String and reuse the string method
    return expr_to_symbolic(string(expr), symbolics)
end

"""
    build_equations(config, symbolics)

Build ModelingToolkit equations from YAML equation strings.
"""
function build_equations(config::Dict, symbolics)

    eqs = Equation[]

    for (state_str, eq_str) in config["equations"]
        state_sym = Symbol(state_str)
        parsed_expr = Meta.parse(eq_str) # parse string
        symbolic_expr = expr_to_symbolic(parsed_expr, symbolics)
        push!(eqs, ModelingToolkit.D(symbolics.states[state_sym]) ~ symbolic_expr)
    end

    return eqs

end 

# -------------------------------------------------------------------------
# Model Info Extraction
# -------------------------------------------------------------------------

"""
    get_model_info(config)

Read model name, description, and type from YAML data.

The information is copied into `ModelDefinition` and is useful for generated
documentation, logging, and later provenance of inference or design results.
"""
function get_model_info(config::Dict)

    exp_cfg = get(config, "experiment", Dict())
    model_cfg = get(config, "model", Dict())

    model_name = get(exp_cfg, "name", "UnnamedModel")
    model_description = get(exp_cfg, "description", "No description provided")
    model_type = Symbol(get(model_cfg, "type", "ODE"))
    
    return (model_name=model_name, 
            model_description=model_description, 
            model_type=model_type)

end

# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------

"""
    load_model(filename::String) -> ModelDefinition

Load a model definition from YAML.

This is the public entry point for the YAML stage. It reads the file with
`load_YAML`, validates the required model sections, builds ModelingToolkit
states and parameters, converts equation strings to symbolic equations, and
returns a `ModelDefinition`.

The YAML file describes the model structure:

- `experiment`: name and description.
- `model`: model type and state names.
- `parameters`: fixed, uncertain, and design parameter metadata.
- `equations`: right-hand sides for each state equation.
- `inputs`: currently a step input signal.

The returned definition is not yet an executable ODE problem. Compile it and
wrap it in a `Model`, then use `SimulationSpec`, `simulate!`, `TuringSpec`, or
`CartesianScanner` for later workflow stages.
"""

function load_model(filename::String)
    config = load_YAML(filename)
    validate_YAML(config)

    info = get_model_info(config)
    syms = build_symbolics(config)
    eqs = build_equations(config, syms)
        

    return ModelDefinition(info.model_name,
                           info.model_description,
                           info.model_type,
                           eqs,
                           syms.states,
                           syms.parameters,
                           syms.input)



end 
