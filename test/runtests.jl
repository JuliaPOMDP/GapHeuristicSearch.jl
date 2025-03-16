using Test
using GapHeuristicSearch
using POMDPs
using POMDPModelTools

# Define a simple test problem
struct SimplePOMDP <: POMDP{Int, Int, Int} end

# Basic required methods for testing
POMDPs.states(::SimplePOMDP) = [1, 2]
POMDPs.actions(::SimplePOMDP) = [1, 2]
POMDPs.observations(::SimplePOMDP) = [1, 2]
POMDPs.initialstate(::SimplePOMDP) = Deterministic(1)
POMDPs.discount(::SimplePOMDP) = 0.9
POMDPs.transition(::SimplePOMDP, s, a) = Deterministic(s)
POMDPs.observation(::SimplePOMDP, a, sp) = Deterministic(a)
POMDPs.reward(::SimplePOMDP, s, a, sp) = 1.0

# Define a basic updater (for testing)
struct SimpleUpdater <: Updater end
POMDPs.initialize_belief(::SimpleUpdater, state) = state_distribution(state=state)
POMDPs.update(::SimplePOMDP, b, a, o) = b

# Begin testing
@testset "GapHeuristicSearch.jl Basic Tests" begin
    
    # Test creating a solver
    solver = GapHeuristicSearchSolver(
        SimpleUpdater(),
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

    @test typeof(solver) == GapHeuristicSearchSolver

    # Test solver parameters
    @test solver.Rmax == -Inf  # Default is -Inf unless explicitly given
    @test solver.delta == 1e-2
    @test solver.d_max == 5

    # Test solving a simple POMDP
    pomdp = SimplePOMDP()
    planner = solve(solver, pomdp)

    @test typeof(planner) == GapHeuristicSearchPlanner{typeof(state_distribution(state=1)), Int, Int}

    # Check get_type
    B, A, O = get_type(planner)
    @test B == typeof(state_distribution(state=1))
    @test A == Int
    @test O == Int

    # Test action function with a simple belief
    b = state_distribution(state=1)
    
    try
    a = action(planner, b)
    @test isa(a, Int)
catch e
    @warn "Action test failed:" exception=(e, catch_backtrace())
    @test false
end

    end
