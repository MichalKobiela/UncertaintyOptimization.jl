using YAML
using ModelingToolkit

const IV = ModelingToolkit.t_nounits
const D = ModelingToolkit.D_nounits

""" 
The aim of this module is to be responsible for reading in a YAML and creating the correct
    structure/format so that it can be used for any type of problem.

    INPUT: YAML
    OUTPUT: Struct
"""

# -------------------------------------------------------------------------
# Struct Definitions
# -------------------------------------------------------------------------

struct ParameterSpec
    name::String # paramater name
    symbol::Any # parameter symbolic
    role:: Symbol # whether :fixed, :uncertain, :design
    value:: Union{Nothing, Float64} # value of the paramater if provided
    bounds::Union{Nothing, Tuple{Float64,Float64}} # bounds for the parameter if provided
    prior:: Union{Nothing, Dict}
end

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

Loads and parses a YAML file. Throws an error if the file does not exist.

"""

function load_YAML(filename:: String)
    if isfile(filename)
        return YAML.load_file(filename)
    else
        println("❌  File with the name $filename not found, please check if the input path is correct and the file exists")
        return nothing
    end
end

"""
    create_param(x)

Helper to convert a char read from the YAML to an @parameter required for the ModellingToolkit system

"""

function create_param(x)
    sym = Symbol(x)
    Symbolics.unwrap(first(@parameters $sym))
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
    validate_YAML(config::Dict)

"""
function validate_YAML(config::Dict)
    # Check the required tags are there
    required_tags = ["experiment", "model", "parameters", "equations"]
    for tag in required_tags
        if !haskey(config, tag)
            println(tag)
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
    
    println("✅ Valid YAML")
    return true

end

# -------------------------------------------------------------------------
# Model Symbolic Construction
# -------------------------------------------------------------------------

function build_symbolics(config::Dict) 

  #  # Symbolic states
    state_symbs = config["model"]["states"] # Read in states from YAML and convert to MTK variable
    state_map = Dict(Symbol(s) => create_var(s, IV) for s in state_symbs)

    # Get parameters specifications
    param_specs = Dict{Symbol, ParameterSpec}()

    for (pname_str, pinfo) in config["parameters"]   
        param = create_param(pname_str)   # create MTK parameter
        role = Symbol(pinfo["role"])
        value = get(pinfo, "value", nothing)
        bounds = haskey(pinfo, "bounds") ? tuple(pinfo["bounds"]...) : nothing
        prior = get(pinfo, "prior", nothing)
        param_specs[Symbol(pname_str)] = ParameterSpec(pname_str, param, role, value, bounds, prior)
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
function expr_to_symbolic(expr, symbolics)
    if expr isa Expr
        # Recursively process each argument of the expression
        return Expr(expr.head, map(arg -> expr_to_symbolic(arg, symbolics), expr.args)...)
    elseif expr isa Symbol
        # Replace with state variable if it's a state
        if haskey(symbolics.states, expr)
            return symbolics.states[expr]
        # Replace with parameter if it's a parameter
        elseif haskey(symbolics.parameters, expr)
            return symbolics.parameters[expr].symbol
        # Replace input variable if it's named :input
        elseif expr == :input
            return symbolics.input
        else
            # Keep as-is if unknown (numbers, functions like sin)
            return expr
        end
    else
        # Numbers and literals are returned as-is
        return expr
    end
end

function build_equations(config::Dict, symbolics)

        eqs = Equation[]

    for (state_str, eq_str) in config["equations"]
        state_sym = Symbol(state_str)
        parsed_expr = Meta.parse(eq_str)                        # parse string to Expr
        symbolic_expr = expr_to_symbolic(parsed_expr, symbolics) # convert symbols directly
        push!(eqs, D(symbolics.states[state_sym]) ~ symbolic_expr)
    end

    return eqs

end 

# -------------------------------------------------------------------------
# Model Info Extraction
# -------------------------------------------------------------------------

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
    load_model_from_yaml(filename::String) -> ModelDefinition

Main entry point: loads, validates, and constructs a full model definition
from a YAML file.

"""

#= function load_model_from_yaml(filename::String)

    config = load_YAML(filename)
    validate_YAML(config)

    info = get_model_info(config)
    syms = build_symbolics(config)
  #  eqs = build_equations(config, syms.states, syms.parameters, syms.input)

   # return #ModelDefinition(info.model_name,
                           info.model_description,
                           info.model_type,
                           eqs,
                           syms.states,
                           syms.parameters,
                           syms.input)



end =#