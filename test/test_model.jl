using Test
using ModelingToolkit
using ModelingToolkit: t_nounits as t
using OrdinaryDiffEq
using DataFrames
include("helpers/mock_rpa.jl")
using .MockRPA

@testset "Warmup setter inputs" begin
    @variables X(t)
    k = UncertaintyOptimization.create_param("k")
    d = UncertaintyOptimization.create_param("d")

    eqs = [
        Differential(t)(X) ~ k * X + d
    ]

    params = Dict{Symbol, UncertaintyOptimization.ParameterSpec}(
        :k => UncertaintyOptimization.ParameterSpec(
            "k",
            k,
            :fixed,
            2.0,
            1.0,
            nothing,
            nothing,
            UncertaintyOptimization.Design(nothing, 3.0),
        ),
        :d => UncertaintyOptimization.ParameterSpec(
            "d",
            d,
            :fixed,
            4.0,
            nothing,
            nothing,
            nothing,
            UncertaintyOptimization.Design(5.0, 6.0),
        ),
    )

    model_def = UncertaintyOptimization.ModelDefinition(
        "WarmupSetterInputs",
        "Tiny model for warmup setter preparation",
        :ODE,
        eqs,
        Dict(:X => X),
        params,
        nothing,
    )

    @mtkcompile warmup_setter_sys = System(model_def.equations, t)
    model = UncertaintyOptimization.Model(model_def, warmup_setter_sys)

    warmup_inputs = UncertaintyOptimization.get_warmup_setter_inputs(model)
    @test warmup_inputs.warmup_values == (1.0,)
    @test Tuple(Symbolics.tosymbol.(warmup_inputs.warmup_Nums)) == (:k,)

    design_warmup_inputs = UncertaintyOptimization.get_warmup_setter_inputs(model; design=true)
    @test design_warmup_inputs.warmup_values == (5.0,)
    @test Tuple(Symbolics.tosymbol.(design_warmup_inputs.warmup_Nums)) == (:d,)
end

@testset "Multiparam setter inputs" begin
    @variables Y(t)
    cuma = UncertaintyOptimization.create_param("cuma")
    fallback = UncertaintyOptimization.create_param("fallback")
    baseline = UncertaintyOptimization.create_param("baseline")
    sampled = UncertaintyOptimization.create_param("sampled"; tunable=true)

    eqs = [
        Differential(t)(Y) ~ cuma * fallback * baseline * sampled * Y
    ]

    params = Dict{Symbol, UncertaintyOptimization.ParameterSpec}(
        :cuma => UncertaintyOptimization.ParameterSpec(
            "cuma",
            cuma,
            :fixed,
            (2e-5, 0.0001, 0.001),
            2e-6,
            nothing,
            nothing,
            UncertaintyOptimization.Design(2e-6, 0.0003),
        ),
        :fallback => UncertaintyOptimization.ParameterSpec(
            "fallback",
            fallback,
            :fixed,
            (10.0, 20.0, 30.0),
            nothing,
            nothing,
            nothing,
            nothing,
        ),
        :baseline => UncertaintyOptimization.ParameterSpec(
            "baseline",
            baseline,
            :fixed,
            7.0,
            nothing,
            nothing,
            nothing,
            nothing,
        ),
        :sampled => UncertaintyOptimization.ParameterSpec(
            "sampled",
            sampled,
            :uncertain,
            5.0,
            nothing,
            nothing,
            Dict("distribution" => "uniform", "lower" => 0.0, "upper" => 10.0),
            nothing,
        ),
    )

    model_def = UncertaintyOptimization.ModelDefinition(
        "MultiparamSetterInputs",
        "Tiny model for multiparam setter preparation",
        :ODE,
        eqs,
        Dict(:Y => Y),
        params,
        nothing,
    )

    @mtkcompile multiparam_setter_sys = System(model_def.equations, t)
    model = UncertaintyOptimization.Model(model_def, multiparam_setter_sys)

    to_stage_maps(inputs) = begin
        symbols = Symbolics.tosymbol.(inputs.multiparam_Nums)
        [Dict(zip(symbols, stage)) for stage in inputs.multiparam_values]
    end

    multiparam_maps = to_stage_maps(UncertaintyOptimization.get_multiparam_setter_inputs(model))
    @test length(multiparam_maps) == 3
    @test multiparam_maps[1][:cuma] == 2e-5
    @test multiparam_maps[2][:cuma] == 0.0001
    @test multiparam_maps[3][:cuma] == 0.001
    @test multiparam_maps[1][:fallback] == 10.0
    @test multiparam_maps[2][:fallback] == 20.0
    @test multiparam_maps[3][:fallback] == 30.0
    @test all(!haskey(stage, :baseline) for stage in multiparam_maps)

    design_multiparam_maps = to_stage_maps(UncertaintyOptimization.get_multiparam_setter_inputs(model; design=true))
    @test length(design_multiparam_maps) == 3
    @test all(stage[:cuma] == 0.0003 for stage in design_multiparam_maps)
    @test design_multiparam_maps[1][:fallback] == 10.0
    @test design_multiparam_maps[2][:fallback] == 20.0
    @test design_multiparam_maps[3][:fallback] == 30.0
    @test all(stage[:baseline] == 7.0 for stage in design_multiparam_maps)
    @test all(!haskey(stage, :sampled) for stage in design_multiparam_maps)
end

@testset "Observed state extraction" begin
    @variables A(t) B(t)
    k = UncertaintyOptimization.create_param("k")

    eqs = [
        Differential(t)(A) ~ k * A,
        Differential(t)(B) ~ 2 * k * B,
    ]

    params = Dict{Symbol, UncertaintyOptimization.ParameterSpec}(
        :k => UncertaintyOptimization.ParameterSpec(
            "k",
            k,
            :fixed,
            1.0,
            nothing,
            nothing,
            nothing,
            nothing,
        ),
    )

    model_def = UncertaintyOptimization.ModelDefinition(
        "ObservedStateExtraction",
        "Tiny model for observed-state index resolution",
        :ODE,
        eqs,
        Dict(:A => A, :B => B),
        params,
        nothing,
    )

    @mtkcompile observed_state_sys = System(model_def.equations, t)

    sim_spec = UncertaintyOptimization.SimulationSpec(
        t_obs = [0.0, 1.0],
        obs_state = [:B, :A],
        initial_conditions = (1.0, 1.0),
        tspan = (0.0, 1.0),
    )
    staged_tspan_spec = UncertaintyOptimization.SimulationSpec(
        t_obs = [0.0, 0.5],
        obs_state = :A,
        initial_conditions = (1.0, 1.0),
        tspan = ((0.0, 10.0), (0.0, 0.5)),
    )

    @test sim_spec.tspan == ((0.0, 1.0), (0.0, 1.0))
    @test staged_tspan_spec.tspan == ((0.0, 10.0), (0.0, 0.5))

    expected_indices = [
        findfirst(isequal(getproperty(observed_state_sys, state)), unknowns(observed_state_sys))
        for state in (:B, :A)
    ]

    @test UncertaintyOptimization.observed_states(sim_spec) == (:B, :A)
    @test UncertaintyOptimization.observed_state_indices(observed_state_sys, sim_spec) == expected_indices
    @test UncertaintyOptimization.observed_state_save_idxs(observed_state_sys, sim_spec) == expected_indices

    sols = [
        [20.0 21.0; 10.0 11.0],
        [40.0 41.0; 30.0 31.0],
    ]

    predicted = UncertaintyOptimization._predicted_observations(sols, sim_spec)
    @test predicted == [20.0, 21.0, 10.0, 11.0, 40.0, 41.0, 30.0, 31.0]

    UncertaintyOptimization._validate_observation_layout(predicted, sim_spec, length(sols))
    @test_throws ErrorException UncertaintyOptimization._validate_observation_layout(predicted[1:end-1], sim_spec, length(sols))
    @test_throws ErrorException UncertaintyOptimization._validate_observation_layout(predicted, sim_spec, length(sols) + 1)
end

@testset "Grid scan default evaluator" begin
    @variables X(t)
    k = UncertaintyOptimization.create_param("k"; tunable=true)
    d = UncertaintyOptimization.create_param("d")

    eqs = [
        Differential(t)(X) ~ k * X + d
    ]

    params = Dict{Symbol, UncertaintyOptimization.ParameterSpec}(
        :k => UncertaintyOptimization.ParameterSpec(
            "k",
            k,
            :uncertain,
            1.0,
            nothing,
            nothing,
            Dict("distribution" => "uniform", "lower" => 0.0, "upper" => 10.0),
            nothing,
        ),
        :d => UncertaintyOptimization.ParameterSpec(
            "d",
            d,
            :fixed,
            3.0,
            nothing,
            nothing,
            nothing,
            nothing,
        ),
    )

    model_def = UncertaintyOptimization.ModelDefinition(
        "CartesianScannerDefaultEvaluator",
        "Tiny model for grid scan default evaluator",
        :ODE,
        eqs,
        Dict(:X => X),
        params,
        nothing,
    )

    @mtkcompile grid_scan_sys = System(model_def.equations, t)
    model = UncertaintyOptimization.Model(model_def, grid_scan_sys)

    sim_spec = UncertaintyOptimization.SimulationSpec(
        t_obs = [0.0, 0.5],
        obs_state = :X,
        initial_conditions = (1.0,),
        tspan = (0.0, 0.5),
        solver = Tsit5(),
    )
    @test sim_spec.obs_state == :X
    @test_throws Exception UncertaintyOptimization.SimulationSpec(
        t_obs = [0.0, 0.5],
        obs_state = 1,
        initial_conditions = (1.0,),
        tspan = (0.0, 0.5),
        solver = Tsit5(),
    )

    manual_sim_model = UncertaintyOptimization.Model(model_def, grid_scan_sys)
    spec_sim_model = UncertaintyOptimization.Model(model_def, grid_scan_sys)
    override_sim_model = UncertaintyOptimization.Model(model_def, grid_scan_sys)

    manual_sols = simulate!(
        manual_sim_model,
        sim_spec.initial_conditions,
        sim_spec.tspan;
        solver=sim_spec.solver,
        saveat=sim_spec.t_obs,
        save_idxs=UncertaintyOptimization.observed_state_save_idxs(manual_sim_model.sys, sim_spec),
        solver_opts=sim_spec.solver_opts,
    )
    spec_sols = simulate!(spec_sim_model, sim_spec)
    override_sols = simulate!(override_sim_model, sim_spec; saveat=[0.5])

    @test Array(only(spec_sols)) ≈ Array(only(manual_sols))
    @test size(Array(only(override_sols)), 2) == 1

    loss_calls = Ref(0)
    loss = function(warmup_sol, predicted_sol; sys=nothing)
        loss_calls[] += 1
        @test sys === model.sys
        @test warmup_sol === nothing
        @test predicted_sol !== nothing
        return sum(Array(predicted_sol))
    end

    scan = UncertaintyOptimization.CartesianScanner(
        simulation = sim_spec,
        scan = [(symbol = :k, values = 1:2, kind = :scale)],
        loss = loss,
    )

    results = UncertaintyOptimization.run_scan([Dict(:k => 1.0)], scan, model)

    @test scan.scan[1].values == [1.0, 2.0]
    @test scan.scan[1].symbol == :k
    @test scan.scan[1].kind == :scale
    @test results isa DataFrame
    @test names(results) == ["iteration", "candidate_index", "k_scaler", "k_value", "loss", "is_best", "best_loss"]
    @test nrow(results) == 2
    @test results.iteration == [1, 1]
    @test results.candidate_index == [1, 2]
    @test results.k_scaler == [1.0, 2.0]
    @test results.k_value == [1.0, 2.0]
    @test count(results.is_best) == 1
    @test all(results.best_loss .== minimum(results.loss))
    @test loss_calls[] == 2

    value_scan = UncertaintyOptimization.CartesianScanner(
        simulation = sim_spec,
        scan = [(symbol = :k, values = [2.0], kind = :value)],
        loss = loss,
    )
    value_results = UncertaintyOptimization.run_scan([Dict(:k => 2.0)], value_scan, model)

    @test value_scan.scan[1].kind == :value
    @test names(value_results) == ["iteration", "candidate_index", "k_value", "loss", "is_best", "best_loss"]
    @test value_results.k_value == [2.0]
    @test only(value_results.is_best)

    scaled_params = [2.0]
    value_params = [2.0]
    UncertaintyOptimization._apply_grid_value!(scaled_params, model, :k, 2.0, :scale)
    UncertaintyOptimization._apply_grid_value!(value_params, model, :k, 2.0, :value)

    @test scaled_params == [4.0]
    @test value_params == [2.0]
    @test_throws ErrorException UncertaintyOptimization.CartesianScanner(
        simulation = sim_spec,
        scan = [(symbol = :k, values = [2.0], kind = :replace)],
        loss = loss,
    )

    combo_scan = UncertaintyOptimization.CartesianScanner(
        simulation = sim_spec,
        scan = [
            (symbol = :k, values = [1.0, 2.0], kind = :scale),
            (symbol = :k, values = [3.0, 4.0], kind = :scale),
        ],
        loss = (values; sys=nothing) -> sum(values),
    )

    combo_results = UncertaintyOptimization.run_scan(
        [Dict(:k => 1.0)],
        combo_scan,
        model;
        evaluator = (_, values, __) -> values,
    )

    @test combo_scan.combinations == [[1.0, 3.0], [1.0, 4.0], [2.0, 3.0], [2.0, 4.0]]
    @test names(combo_results) == ["iteration", "candidate_index", "k_scaler", "k_value", "k_scaler_2", "k_value_2", "loss", "is_best", "best_loss"]
    @test nrow(combo_results) == 4
    @test combo_results.k_scaler == [1.0, 1.0, 2.0, 2.0]
    @test combo_results.k_value == [1.0, 1.0, 2.0, 2.0]
    @test combo_results.k_scaler_2 == [3.0, 4.0, 3.0, 4.0]
    @test combo_results.k_value_2 == [3.0, 4.0, 3.0, 4.0]
    @test count(combo_results.is_best) == 1
    @test only(combo_results.k_scaler[combo_results.is_best]) == 1.0
    @test only(combo_results.k_value[combo_results.is_best]) == 1.0
    @test only(combo_results.k_scaler_2[combo_results.is_best]) == 3.0
    @test only(combo_results.k_value_2[combo_results.is_best]) == 3.0
    @test all(combo_results.best_loss .== minimum(combo_results.loss))

    fixed_scan = UncertaintyOptimization.CartesianScanner(
        simulation = sim_spec,
        scan = [(symbol = :d, values = [2.0], kind = :scale)],
        loss = (warmup_sol, predicted_sol; sys=nothing) -> Array(predicted_sol)[end],
    )

    fixed_results = UncertaintyOptimization.run_scan([Dict(:k => 1.0)], fixed_scan, model)

    @test names(fixed_results) == ["iteration", "candidate_index", "d_scaler", "d_value", "loss", "is_best", "best_loss"]
    @test only(fixed_results.d_value) == 6.0
    @test only(fixed_results.loss) > 5.0
end

@testset "Test Model simulations" begin

    @testset "Test one off simulation" begin
    
        model_def = MockRPA.mock_rpa_model()

        @mtkcompile sys = System(model_def.equations, t)

        model = UncertaintyOptimization.Model(model_def, sys)

        u0 = (1.0, 1.0)

        params = Dict(
            :beta_RA => 0.1,
            :beta_AB => 0.001,
            :beta_BA => 0.01,
            :beta_BB => 0.001
        )
            
        tspan = (0.0, 100.0) 
            
        # Run simulation
        sol = simulate!(model, u0, tspan; parameters=params)
            
        # Check that solution exists
        @test sol !== nothing

        sol_with_warmup = simulate!(model, u0, tspan; parameters=params, return_simulate=true)
        @test sol_with_warmup.warmup_sol === nothing
        @test sol_with_warmup.sols !== nothing

    end

    @testset "Test setup simulation" begin
    
        model_def = MockRPA.mock_rpa_model()

        @mtkcompile sys = System(model_def.equations, t)

        model = UncertaintyOptimization.Model(model_def, sys)

        t_obs = collect(range(0.0, 100.0, length=10))
        params = Dict(
            :beta_RA => 0.1,
            :beta_AB => 0.001,
            :beta_BA => 0.01,
            :beta_BB => 0.001
        )
        
        tspan = (0.0, 100.0)

        UncertaintyOptimization.setup_simulation!(
            model,
            (1.0, 1.0),
            tspan;
            parameters=params,
        )

        @test model.prob !== nothing

        # Call simulate with uncertain params.
        predicted_sol = simulate!(
            model,
            (1.0, 1.0),
            tspan;
            solver=Euler(),
            solver_opts=(dt=0.01,),
            saveat=t_obs,
            save_idxs=1,
            sampled_uncertain_params=[0.1, 0.1, 0.1, 0.1],
        )[1]

        predicted = Array(predicted_sol)
        
        @test length(predicted) == length(t_obs)
        @test all(isfinite, predicted)
        

    end

    @testset "RPA default snapshot regression" begin
        model_def = MockRPA.mock_rpa_model()

        @mtkcompile rpa_snapshot_sys = System(model_def.equations, t)
        model = UncertaintyOptimization.Model(model_def, rpa_snapshot_sys)

        default_parameters = Dict{Symbol, Float64}(
            name => Float64(spec.value)
            for (name, spec) in model_def.parameters
            if spec.value isa Real
        )

        expected_default_parameters = Dict{Symbol, Float64}(
            :alpha_1 => 100.0,
            :alpha_2 => 100.0,
            :beta_RA => 0.0,
            :beta_AB => 0.0,
            :beta_BA => 0.0,
            :beta_BB => 0.0,
            :gamma_A => 1.0,
            :gamma_B => 1.0,
            :n_RA => 1.0,
            :n_BA => 1.0,
            :n_AB => 1.0,
            :n_BB => 1.0,
            :K_IR => 1.0,
            :K_TF => 1.0,
            :K_BA => 1.0,
            :K_AB => 1.0,
            :K_BB => 1.0,
        )

        t_obs = [0.0, 25.0, 50.0, 75.0, 100.0]
        sol = only(simulate!(
            model,
            (1.0, 1.0),
            (0.0, 100.0);
            parameters=default_parameters,
            solver=Euler(),
            solver_opts=(dt=1.0,),
            saveat=t_obs,
        ))

        expected = [
            1.0 1.4630413659109438 1.4630541869912255 1.0683276764072513 1.0683453199952995;
            1.0 39.59947289771795 39.599999995572006 47.346492679778905 47.347826043084
        ]

        # Exact equality is intentional: this is a golden snapshot for default behavior.
        @test default_parameters == expected_default_parameters
        @test model.tunable_symbols == (:beta_BA, :beta_AB, :beta_RA, :beta_BB)
        @test model.tunable_initial == (0.0, 0.0, 0.0, 0.0)
        @test Array(sol) == expected
    end

    @testset "RPA real YAML default snapshot regression" begin
        yaml_file = joinpath(@__DIR__, "test-data", "RPA_real", "opt.yml")
        model_def = load_model(yaml_file)

        @mtkcompile rpa_real_yaml_snapshot_sys = System(model_def.equations, t)
        model = UncertaintyOptimization.Model(model_def, rpa_real_yaml_snapshot_sys)

        sim = simulate!(
            model,
            (24.0, 350.0),
            (0.0, 10.0);
            solver=Euler(),
            solver_opts=(dt=0.01,),
            saveat=[0.0, 2.5, 5.0, 7.5, 10.0],
            return_simulate=true,
        )

        expected_solutions = [
            [
                3.381524467028881 8.192509924511999 7.969672732275744 6.767388503907415 5.920390469734003;
                6580.152516341223 1520.9485393023892 323.6212931732738 93.35172926643642 72.992523912589
            ],
            [
                3.381524467028881 24.498424541094924 23.29912895394998 15.340392550902282 8.120004783250028;
                6580.152516341223 1182.8023380555533 205.04816021322162 36.97812122415498 21.233364834530857
            ],
            [
                3.381524467028881 26.326602510304415 25.030613391501756 16.403515692637164 8.23990952415055;
                6580.152516341223 1178.1462001008592 203.9430102842959 36.468064341419606 19.843743627053442
            ],
        ]

        @test model.warmup_values == (2.0e-6,)
        @test model.multiparam_values == ((2.0e-5,), (0.0001,), (0.001,))
        @test model.tunable_symbols == (:alpha_3, :beta_3, :beta_2, :nx1, :alpha_4, :nx2, :alpha_2, :kx2, :beta_1, :r1, :kx1, :r2, :kr, :alpha_1, :nr, :beta_4)
        @test model.tunable_initial == (17.7437, 0.6644, 0.00039, 2.34, 8.7519e6, 1.3, 25.0, 36.4063, 11.9586, 89.0635, 1.28e-8, 7.0188, 0.51, 83.4743, 3.2, 7.1347)
        @test sim.warmup_sol.u[end] == [3.381524467028881, 6580.152516341223]
        @test [Array(sol) for sol in sim.sols] == expected_solutions
    end

    @testset "Test multiparam simulation" begin
        @variables Y(t)
        k = UncertaintyOptimization.create_param("k")

        eqs = [
            Differential(t)(Y) ~ k
        ]

        params = Dict{Symbol, UncertaintyOptimization.ParameterSpec}(
            :k => UncertaintyOptimization.ParameterSpec(
                "k",
                k,
                :fixed,
                (1.0, 2.0, 3.0),
                0.0,
                nothing,
                nothing,
                nothing,
            ),
        )

        model_def = UncertaintyOptimization.ModelDefinition(
            "MultiparamSimulation",
            "Tiny model for validating isolated multiparam solves",
            :ODE,
            eqs,
            Dict(:Y => Y),
            params,
            nothing,
        )

        @mtkcompile multiparam_simulation_sys = System(model_def.equations, t)
        model = UncertaintyOptimization.Model(model_def, multiparam_simulation_sys)

        sols = simulate!(
            model,
            (0.0,),
            (0.0, 1.0);
            solver=Euler(),
            solver_opts=(dt=0.1,),
            saveat=[1.0],
        )

        @test length(sols) == 3
        @test [Array(sol)[end] for sol in sols] ≈ [1.0, 2.0, 3.0]

        staged_tspan_sim = simulate!(
            model,
            (0.0,),
            ((0.0, 1.0), (0.0, 0.5));
            solver=Euler(),
            solver_opts=(dt=0.1,),
            saveat=[0.5],
            return_simulate=true,
        )

        @test staged_tspan_sim.warmup_sol.t[end] == 1.0
        @test all(sol.t[end] == 0.5 for sol in staged_tspan_sim.sols)
        @test [Array(sol)[end] for sol in staged_tspan_sim.sols] ≈ [0.5, 1.0, 1.5]
    end
    
end

# @testset "Test Model constructor" begin

#     config = Dict(  
#         "parameters" => Dict(
#             "alpha" => Dict("role"=>"fixed","value"=>1.0),
#             "beta"  => Dict("role"=>"uncertain","value"=>0.0),
#             "gamma" => Dict("role"=>"design","value"=>0.1)
#         ),
#         "model" => Dict("states" => ["X", "Y"]),
#         "inputs" => Dict("type" => "step", "t_threshold"=>5.0, "values"=>[0.0,1.0]),
#         "equations" => Dict(
#             "X" => "alpha*X + beta*Y - gamma*X",
#             "Y" => "beta*X - gamma*Y*input"
#         )
#     )

#     info = UncertaintyOptimization.get_model_info(config)
#     syms = UncertaintyOptimization.build_symbolics(config)
#     eqs = UncertaintyOptimization.build_equations(config, syms)

#     model_def = UncertaintyOptimization.ModelDefinition(info.model_name,
#                            info.model_description,
#                            info.model_type,
#                            eqs,
#                            syms.states,
#                            syms.parameters,
#                            syms.input)

#     @test typeof(model_def) == ModelDefinition
#     @mtkcompile sys = System(model_def.equations, t)
        
#     # Create the Model object
#     model = UncertaintyOptimization.Model(model_def, sys)
        
#     # Test that Model was created with correct fields
#     @test typeof(model) == Model
#     @test model.model_def == model_def
#     @test model.sys == sys
        
#     # Test that prob and sol are initially nothing
#     @test model.prob === nothing
#     @test model.sol === nothing
        
    
# end

# @testset "Model simulate! Tests" begin
    
#     # Setup - create a model for all tests
#     filename = joinpath(@__DIR__, "test-data", "test_RPA.yml")
#     model_def = UncertaintyOptimization.load_model_from_yaml(filename)
#     @mtkcompile sys1 = System(model_def.equations, t)
#     model = UncertaintyOptimization.Model(model_def, sys1)

#     # Define simulation parameters
#     init_cond = [1.0, 1.0]  # Initial conditions for [R, A]
        
#     # Parameters to simulate with (ground truth values)
#     params = Dict(
#         :beta_RA => 0.1,
#         :beta_AB => 0.001,
#         :beta_BA => 0.01,
#         :beta_BB => 0.001
#     )
        
#     tspan = (0.0, 100.0)  # Simulate from t=0 to t=10
        
#     # Run simulation
#     sol = simulate!(model, init_cond, params, tspan)
        
#     # Check that solution exists
#     @test sol !== nothing
        
#     # Check that model.sol was updated
#     @test model.sol === sol
        
#     # Check that model.prob was created
#     @test model.prob !== nothing
        


# end
