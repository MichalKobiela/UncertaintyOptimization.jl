
module MockRPA

using ModelingToolkit
using UncertaintyOptimization: ModelDefinition, ParameterSpec

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
    @parameters alpha_1 alpha_2
    @parameters beta_RA beta_AB beta_BA beta_BB
    @parameters gamma_A gamma_B
    @parameters n_RA n_BA n_AB n_BB
    @parameters K_IR K_TF K_BA K_AB K_BB

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

    # design params
    params[:alpha_1] = ParameterSpec("alpha_1", alpha_1, :design, 100.0, (10.0,200.0), nothing)
    params[:alpha_2] = ParameterSpec("alpha_2", alpha_2, :design, 100.0, (10.0,200.0), nothing)
    params[:K_BA]    = ParameterSpec("K_BA",    K_BA,    :design, 1.0, (0.1,10.0), nothing)
    params[:K_AB]    = ParameterSpec("K_AB",    K_AB,    :design, 1.0, (0.1,10.0), nothing)
    params[:K_BB]    = ParameterSpec("K_BB",    K_BB,    :design, 1.0, (0.1,10.0), nothing)

    # uncertain params with priors
    uniform_prior = Dict("distribution"=>"uniform","lower"=>0.0,"upper"=>1.0)
    params[:beta_RA] = ParameterSpec("beta_RA", beta_RA, :uncertain, 0.0, nothing, uniform_prior)
    params[:beta_AB] = ParameterSpec("beta_AB", beta_AB, :uncertain, 0.0, nothing, uniform_prior)
    params[:beta_BA] = ParameterSpec("beta_BA", beta_BA, :uncertain, 0.0, nothing, uniform_prior)
    params[:beta_BB] = ParameterSpec("beta_BB", beta_BB, :uncertain, 0.0, nothing, uniform_prior)

    # fixed params
    params[:gamma_A] = ParameterSpec("gamma_A", gamma_A, :fixed, 1.0, nothing, nothing)
    params[:gamma_B] = ParameterSpec("gamma_B", gamma_B, :fixed, 1.0, nothing, nothing)
    params[:n_RA]    = ParameterSpec("n_RA",    n_RA,    :fixed, 1.0, nothing, nothing)
    params[:n_BA]    = ParameterSpec("n_BA",    n_BA,    :fixed, 1.0, nothing, nothing)
    params[:n_AB]    = ParameterSpec("n_AB",    n_AB,    :fixed, 1.0, nothing, nothing)
    params[:n_BB]    = ParameterSpec("n_BB",    n_BB,    :fixed, 1.0, nothing, nothing)
    params[:K_IR]    = ParameterSpec("K_IR",    K_IR,    :fixed, 1.0, nothing, nothing)
    params[:K_TF]    = ParameterSpec("K_TF",    K_TF,    :fixed, 1.0, nothing, nothing)

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
