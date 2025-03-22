using Test
using GapHeuristicSearch
using POMDPs
using POMDPModelTools
using POMDPPolicies
using POMDPSimulators

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

# Basic updater for testing
struct SimpleUpdater <: Updater end
POMDPs.initialize_belief(::SimpleUpdater, state) = Deterministic(state)
POMDPs.update(::SimpleUpdater, b, a, o) = b

@testset "GapHeuristicSearch.jl Basic Tests" begin

    # Test solver initialization
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

    # Test solver parameters explicitly
    @test solver.Rmax == 10.0
    @test solver.delta == 0.01
    @test solver.d_max == 5

    # Test planner initialization
    pomdp = SimplePOMDP()
    planner = solve(solver, pomdp)
    @test isa(planner, GapHeuristicSearchPlanner)

    # Ensure correct belief type
    B, A, O = get_type(planner)
    @test B <: Deterministic
    @test A == Int
    @test O == Int

    # Test action method explicitly
    try
        b = Deterministic(1)
        a = action(planner, b)
        @test isa(a, Int)
    catch e
        @warn "Action function failed" exception=(e, catch_backtrace())
        @test false
    end
end
