using Test
using GapHeuristicSearch
using POMDPs
using POMDPTools

# Define a simple test problem
struct SimplePOMDP <: POMDP{Int, Int, Int} end

# Required methods
POMDPs.states(::SimplePOMDP) = [1, 2]
POMDPs.actions(::SimplePOMDP) = [1, 2]
POMDPs.observations(::SimplePOMDP) = [1, 2]
POMDPs.initialstate(::SimplePOMDP) = Deterministic(1)
POMDPs.discount(::SimplePOMDP) = 0.9
POMDPs.transition(::SimplePOMDP, s, a) = Deterministic(s)
POMDPs.observation(::SimplePOMDP, a, sp) = Deterministic(a)
POMDPs.reward(::SimplePOMDP, s, a, sp) = 1.0

# Simple updater for testing
struct SimpleUpdater <: Updater end
POMDPs.initialize_belief(::SimpleUpdater, state) = Deterministic(state)
POMDPs.update(::SimpleUpdater, b, a, o) = b

@testset "GapHeuristicSearch.jl Basic Tests" begin

    # Test Solver Initialization
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

    # Check parameters explicitly
    @test solver.Rmax == 10.0
    @test solver.delta == 0.01
    @test solver.d_max == 5

    # Testing solving the simple problem
    pomdp = SimplePOMDP()

    planner = nothing
    @testset "Solver Test" begin
        try
            planner = solve(solver, pomdp)
            @test isa(planner, GapHeuristicSearchPlanner)
        catch e
            @warn "Solver test failed with error:" exception=(e, catch_backtrace())
            @test false
        end
    end

    # Check get_type function
    if planner !== nothing
        B, A, O = get_type(planner)
        @test B == Deterministic{Int}
        @test A == Int
        @test O == Int
    else
        @warn "Planner not initialized; skipping get_type tests."
        @test false
    end

    # Action testing
    @testset "Action Test" begin
        b = Deterministic(1)
        try
            a = action(planner, b)
            @test isa(a, Int)
        catch e
            @warn "Action test failed with error: $e"
            @test false
        end
    end
end
