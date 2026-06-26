
module MockRPA

using ModelingToolkit
using UncertaintyOptimization: ModelDefinition, ParameterSpec, create_param

const IV = ModelingToolkit.t_nounits 


"""
mock_rpa_model()

This avoids parsing YAML in tests while providing identical symbolic objects.
"""
function mock_rpa_model()
    # symbolic independent variable
    t = IV

    # states
    @variables A(t) B(t)

    # parameters (symbols)
    alpha_1 = create_param("alpha_1")
    alpha_2 = create_param("alpha_2")
    beta_RA = create_param("beta_RA"; tunable=true)
    beta_AB = create_param("beta_AB"; tunable=true)
    beta_BA = create_param("beta_BA"; tunable=true)
    beta_BB = create_param("beta_BB"; tunable=true)
    gamma_A = create_param("gamma_A")
    gamma_B = create_param("gamma_B")
    n_RA = create_param("n_RA")
    n_BA = create_param("n_BA")
    n_AB = create_param("n_AB")
    n_BB = create_param("n_BB")
    K_IR = create_param("K_IR")
    K_TF = create_param("K_TF")
    K_BA = create_param("K_BA")
    K_AB = create_param("K_AB")
    K_BB = create_param("K_BB")

    # input step function (symbolic)
    input_expr = ifelse(t < 50, 1.0, 10.0)

    # A: alpha_1*(1/(1+(K_TF/(1+(input/K_IR)))^n_RA) + beta_RA)*(1/((K_BA/B)^n_BA + 1) + beta_BA) - gamma_A*A
    rhs_A = alpha_1 * ( 1/(1 + (K_TF/(1 + (input_expr/K_IR)))^n_RA) + beta_RA ) *
                  ( 1/((K_BA / B)^n_BA + 1) + beta_BA ) - gamma_A * A

    # B: alpha_2*(1/(1+(A/K_AB)^n_AB) + beta_AB)*(1/((K_BB/B)^n_BB + 1) + beta_BB) - gamma_B*B
    rhs_B = alpha_2 * ( 1/(1 + (A / K_AB)^n_AB) + beta_AB ) *
                  ( 1/((K_BB / B)^n_BB + 1) + beta_BB ) - gamma_B * B

    eqs = [
        Differential(t)(A) ~ rhs_A,
        Differential(t)(B) ~ rhs_B
    ]

    # Build ParameterSpec dict reflecting YAML (role, values, bounds, prior dict)
    params = Dict{Symbol, ParameterSpec}()
    parameter_spec(args...) = ParameterSpec(args..., nothing)

    # design params
    params[:alpha_1] = parameter_spec("alpha_1", alpha_1, :design, 100.0, nothing, (10.0,200.0), nothing)
    params[:alpha_2] = parameter_spec("alpha_2", alpha_2, :design, 100.0, nothing, (10.0,200.0), nothing)
    params[:K_BA]    = parameter_spec("K_BA",    K_BA,    :design, 1.0, nothing, (0.1,10.0), nothing)
    params[:K_AB]    = parameter_spec("K_AB",    K_AB,    :design, 1.0, nothing, (0.1,10.0), nothing)
    params[:K_BB]    = parameter_spec("K_BB",    K_BB,    :design, 1.0, nothing, (0.1,10.0), nothing)

    # uncertain params with priors
    uniform_prior = Dict("distribution"=>"uniform","lower"=>0.0,"upper"=>1.0)
    params[:beta_RA] = parameter_spec("beta_RA", beta_RA, :uncertain, 0.0, nothing, nothing, uniform_prior)
    params[:beta_AB] = parameter_spec("beta_AB", beta_AB, :uncertain, 0.0, nothing, nothing, uniform_prior)
    params[:beta_BA] = parameter_spec("beta_BA", beta_BA, :uncertain, 0.0, nothing, nothing, uniform_prior)
    params[:beta_BB] = parameter_spec("beta_BB", beta_BB, :uncertain, 0.0, nothing, nothing, uniform_prior)

    # fixed params
    params[:gamma_A] = parameter_spec("gamma_A", gamma_A, :fixed, 1.0, nothing, nothing, nothing)
    params[:gamma_B] = parameter_spec("gamma_B", gamma_B, :fixed, 1.0, nothing, nothing, nothing)
    params[:n_RA]    = parameter_spec("n_RA",    n_RA,    :fixed, 1.0, nothing, nothing, nothing)
    params[:n_BA]    = parameter_spec("n_BA",    n_BA,    :fixed, 1.0, nothing, nothing, nothing)
    params[:n_AB]    = parameter_spec("n_AB",    n_AB,    :fixed, 1.0, nothing, nothing, nothing)
    params[:n_BB]    = parameter_spec("n_BB",    n_BB,    :fixed, 1.0, nothing, nothing, nothing)
    params[:K_IR]    = parameter_spec("K_IR",    K_IR,    :fixed, 1.0, nothing, nothing, nothing)
    params[:K_TF]    = parameter_spec("K_TF",    K_TF,    :fixed, 1.0, nothing, nothing, nothing)

    states = Dict(:A => A, :B => B)

    return ModelDefinition(
        "RPA",
        "A two gene robust perfect adaptation circuit to test that things work",
        :ODE,
        eqs,
        states,
        params,
        input_expr
    )
end

end 
