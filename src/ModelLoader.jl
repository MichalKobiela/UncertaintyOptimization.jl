using YAML
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D;

""" 
The aim of this module is to be responsible for reading in a YAML and creating the correct
    structure/format so that it can be used for any type of problem.

    INPUT: YAML
    OUTPUT: Struct
"""

# Struct for each parameter
struct ParameterSpec
    name::String # paramater name
    symbol::Symbol # parameter symbolic
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
    parameters::Dict{Symbol, ParameterSpec}
    input::Any
end

function load_YAML(filename:: String)
    if isfile(filename)
        return YAML.load_file(filename)
    else
        println("❌  File with the name $filename not found, please check if the input path is correct and the file exists")
        return nothing
    end
end

# Builds a Dict of symbolic forms of YAML inputs to be used later
function build_symbolics(config::Dict) 

    # Symbolic states
    state_symbs = Symbol.(config["model"]["states"]) # Read in states from YAML and convert to Julia symbol
    state_map = Dict(s => eval(:(@variables $(s)(t)))[1] for s in state_symbs)

    # Get parameters specifications
    param_specs = Dict{Symbol, ParameterSpec}()
    for (pname_str, pinfo) in config["parameters"]   
        sym = Symbol(pname_str)      
        role = Symbol(pinfo["role"])
        value = get(pinfo, "value", nothing)
        bounds = haskey(pinfo, "bounds") ? tuple(pinfo["bounds"]...) : nothing
        prior = get(pinfo, "prior", nothing)
        param_specs[Symbol(pname_str)] = ParameterSpec(pname_str, sym, role, value, bounds, prior)
    end

    # Makes an input signal defined by the YAML
    if config["inputs"]["type"] == "step"
        input = ifelse(t < config["inputs"]["t_threshold"],
                config["inputs"]["values"][1],
                config["inputs"]["values"][2])
    else
        error("❌ Unsupported input signal type: $(input_cfg["type"])")
    end

    return (states=state_map, parameters=param_specs, input=input)

end

# Build an equation but also handle any missing paramaters or mismatched/unused parameters

function build_equations(config::Dict, state_map::Dict, param_specs::Dict, input)

    eqs = []
    all_symbolics = Dict(s.symbol => s.symbol for s in values(param_specs))
    all_symbolics = merge(all_symbolics, state_map)
    if input !== nothing
        all_symbolics[:input] = input
    end
    
    used_params = Set{Symbol}() # Used to check that all parameters are being used in the equations

    for (state_str, expr_str) in config["equations"]

        state_sym = Symbol(state_str)
        expr = Meta.parse(expr_str) # parse string into an expression

     try
            is_defined = [p for p in keys(all_symbolics) if occursin(string(p), expr_str)]
            union!(used_params, is_defined) # check that all parameters are being used in the equation

            full_exp = Expr(:block, [
                :( $(k) = $(v) ) for (k,v) in all_symbolics if occursin(string(k), expr_str)
            ]..., expr)
            expr_sym = eval(full_exp)

            push!(eqs,D(all_symbolics[state_sym]) ~ expr_sym)
     catch e
            if isa(e, UndefVarError)
              println("❌  Parameter and Equation mismatch: equation for $state_sym references unknown parameters: $(e.var). Please check that all parameters are defined for your equations in your YAML")
          else
              rethrow(e)
          end
      end
    end

    declared_params = Set(keys(all_symbolics))
    unused_params = setdiff(declared_params, used_params)

    # Warn that not all parameters are used
    if !isempty(unused_params)
        println("⚠️  Warning: Unused parameters in YAML: $(collect(unused_params))")
    end

    return eqs

end

function get_model_info(config::Dict)

    exp_cfg = get(config, "experiment", Dict())
    model_cfg = get(config, "model", Dict())

    model_name = get(exp_cfg, "name", "UnnamedModel")
    model_description = get(exp_cfg, "description", "No description provided")
    model_type = Symbol(get(model_cfg, "type", "ODE"))
    return (model_name=model_name, 
            model_description=model_description, 
            model_type=model_type, 
            u0=u0,
            tspan=tspan)

end

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
        if !(state in model_cfg["states"])
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

