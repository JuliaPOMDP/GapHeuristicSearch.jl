using Test
using POMDPs
using POMDPModelTools
using POMDPPolicies
using POMDPSimulators
using Random
using GapHeuristicSearch

# Define a simple test POMDP for validation
struct SimplePOMDP <: POMDP{Int, Int, Int} end

POMDPs.states(::SimplePOMDP) = [1, 2, 3]
POMDPs.actions(::SimplePOMDP) = [1, 2]
POMDPs.observations(::SimplePOMDP) = [1, 2]
POMDPs.transition(::SimplePOMDP, s, a) = Deterministic(s == 1 ? 2 : 3)
POMDPs.observation(::SimplePOMDP, s, a, sp) = Deterministic(sp == 2 ? 1 : 2)
POMDPs.reward(::SimplePOMDP, s, a) = s == 1 ? 10.0 : 0.0
POMDPs.discount(::SimplePOMDP) = 0.9
POMDPs.initialstate(::SimplePOMDP) = Deterministic(1)

# Define a basic updater
struct SimpleUpdater <: Updater end
POMDPs.initialize_belief(::SimpleUpdater, d) = Deterministic(d)
POMDPs.update(::SimpleUpdater, b, a, o) = Deterministic(o)

@testset "GapHeuristicSearch Tests" begin
    pomdp = SimplePOMDP()
    updater = SimpleUpdater()
    solver = GapHeuristicSearchSolver(updater; Rmax=10.0, k_max=10, d_max=5, verbose=false)
    planner = solve(solver, pomdp)
    
    @test planner isa GapHeuristicSearchPlanner
    
    belief = initialize_belief(updater, initialstate(pomdp))
    action = action(planner, belief)
    
    @test action in actions(pomdp)
    
    println("All tests passed!")
end
