using Test
using GapHeuristicSearch
using POMDPs
using POMDPTools

# Define simple test POMDP type
struct SimplePOMDP <: POMDP{Int, Int, Int} end

# Required basic methods for SimplePOMDP
POMDPs.states(::SimplePOMDP) = [1, 2]
POMDPs.actions(::SimplePOMDP) = [1, 2]
POMDPs.observations(::SimplePOMDP) = [1, 2]
POMDPs.initialstate(::SimplePOMDP) = Deterministic(1)
POMDPs.discount(::SimplePOMDP) = 0.9
POMDPs.transition(::SimplePOMDP, s, a) = Deterministic(s)
POMDPs.observation(::SimplePOMDP, a, sp) = Deterministic(a)
POMDPs.reward(::SimplePOMDP, s, a, sp) = 1.0

# Updated basic updater for testing clearly without nested Deterministic
struct SimpleUpdater <: Updater end
POMDPs.initialize_belief(::SimpleUpdater, state) = state  # corrected line
POMDPs.update(::SimpleUpdater, b, a, o) = b

@testset "GapHeuristicSearch.jl Basic Tests" begin

    # Solver initialization test
    solver = GapHeuristicSearchSolver(
        SimpleUpdater();
        π = nothing,
        Rmax = 10.0,
        uhi_func = (pomdp, b) -> 10.0,
        ulo_func = (pomdp, b) -> 0.0,
        delta = 0.01,
        k_max = 10,
        d_max = 5,
        nsamps = 10,
        max_steps = 10
    )

    @test isa(solver, GapHeuristicSearchSolver)

    # Solver parameters clearly checked
    @test solver.Rmax == 10.0
    @test solver.delta == 0.01
    @test solver.d_max == 5

    # Solve simple POMDP explicitly
    planner = nothing
    try
        pomdp = SimplePOMDP()
        planner = solve(solver, pomdp)
        @test isa(planner, GapHeuristicSearchPlanner)
    catch e
        @warn "Solver failed: $e" exception=(e, catch_backtrace())
        @test false
    end

    # Check planner type consistency and action function
    if planner !== nothing
        # Test get_type
        B, A, O = get_type(planner)
        @test B == Deterministic{Int}
        @test A == Int
        @test O == Int

        # Corrected action test without nested Deterministic issue
        b = Deterministic(1)
        try
            a = action(planner, b)
            @test isa(a, Int)
        catch e
            @warn "Action function failed: $e" exception=(e, catch_backtrace())
            @test false
        end
    else
        @warn "Planner initialization failed, skipping further tests."
    end
end
