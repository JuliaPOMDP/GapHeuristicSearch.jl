using Test
using POMDPs
using POMDPModelTools
using POMDPPolicies
using POMDPSimulators
using Random 
using Test

# Import the module (assume it is in the same directory or included)
include("GapHeuristicSearch.jl")
using .GapHeuristicSearch

# Define a Simple POMDP for testing
struct SimplePOMDP <: POMDP{Int, Int, Int} end

POMDPs.actions(::SimplePOMDP) = [1, 2]  # Two actions available
POMDPs.discount(::SimplePOMDP) = 0.95

# Define transition model
function POMDPs.transition(pomdp::SimplePOMDP, s::Int, a::Int)
    if a == 1
        return SparseCat([1, 2], [0.8, 0.2])
    else
        return SparseCat([1, 2], [0.5, 0.5])
    end
end

# Define reward model
function POMDPs.reward(pomdp::SimplePOMDP, s::Int, a::Int)
    return a == 1 ? 10.0 : 5.0
end

# Define observation model
function POMDPs.observation(pomdp::SimplePOMDP, a::Int, sp::Int)
    return SparseCat([1, 2], [0.7, 0.3])
end

# Define initial state distribution
POMDPs.initialstate(::SimplePOMDP) = SparseCat([1, 2], [0.5, 0.5])

# Define a trivial belief updater
struct SimpleUpdater <: Updater end

function POMDPs.initialize_belief(::SimpleUpdater, d)
    return Dict(1 => 0.5, 2 => 0.5)
end

function POMDPs.update(::SimpleUpdater, b, a, o)
    return b  # Just return the same belief for simplicity
end

# Run Tests
@testset "GapHeuristicSearch.jl Tests" begin
    rng = Random.default_rng()
    pomdp = SimplePOMDP()
    up = SimpleUpdater()

    solver = GapHeuristicSearchSolver(
        up,
        π=nothing,  # No rollout policy
        Rmax=10.0,
        uhi_func=nothing,
        ulo_func=nothing,
        delta=1e-2,
        k_max=50,
        d_max=5,
        nsamps=10,
        max_steps=10,
        verbose=false
    )

    planner = solve(solver, pomdp)
    
    # Check if the planner object was created
    @test typeof(planner) <: GapHeuristicSearchPlanner

    # Check if an action is selected
    action_selected = action(planner, initialize_belief(up, initialstate(pomdp)))
    @test action_selected in actions(pomdp)

    println("All tests passed!")
end
