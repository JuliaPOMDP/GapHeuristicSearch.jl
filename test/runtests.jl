using Test
using POMDPs
using POMDPModelTools
using POMDPPolicies
using POMDPSimulators
using Random
using GapHeuristicSearch

# Define a simple POMDP model for testing
struct TestPOMDP <: POMDPs.POMDP{Int, Int, Int} end

POMDPs.states(::TestPOMDP) = 1:2
POMDPs.actions(::TestPOMDP) = [1, 2]
POMDPs.observations(::TestPOMDP) = [1, 2]
POMDPs.transition(::TestPOMDP, s, a) = SparseCat([1, 2], [0.5, 0.5])
POMDPs.observation(::TestPOMDP, s′, a) = SparseCat([1, 2], [0.5, 0.5])
POMDPs.reward(::TestPOMDP, s, a) = 1.0
POMDPs.initialstate(::TestPOMDP) = SparseCat([1, 2], [0.5, 0.5])
POMDPs.discount(::TestPOMDP) = 0.9

@testset "GapHeuristicSearch Tests" begin
    # Create a test POMDP
    pomdp = TestPOMDP()

    # Define a basic updater and policy
    struct TestUpdater <: Updater end
    struct TestPolicy <: Policy end

    POMDPs.update(::TestUpdater, b, a, o) = b
    POMDPs.action(::TestPolicy, b) = 1

    up = TestUpdater()
    π = TestPolicy()

    # Create the solver
    solver = GapHeuristicSearchSolver(up, π=π, Rmax=10.0, delta=0.01, k_max=100, d_max=5, nsamps=10, max_steps=50, verbose=false)

    # Solve the POMDP
    planner = solve(solver, pomdp)

    # Ensure the solver returns a valid policy
    @test isa(planner, GapHeuristicSearchPlanner)
    
    # Test action selection
    action_result = action(planner, initialstate(pomdp))
    @test action_result in actions(pomdp)
end
