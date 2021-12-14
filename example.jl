using POMDPs
using POMDPPolicies
using POMDPSimulators
using POMDPModels
using BeliefUpdaters

push!(LOAD_PATH,"/Users/ericlux/Desk/GHS.jl")
using GHS

cryingbaby = BabyPOMDP()
roller = RandomPolicy(cryingbaby)
up = DiscreteUpdater(cryingbaby)
Rmax = 0.0  # best possible reward is baby not hungry, didnt feed
solver = GapHeuristicSearchSolver(roller,
                                up,
                                Rmax,
                                delta=.1,
                                k_max=500,
                                d_max=7,
                                nsamps=20,
                                max_steps=10)

ghs_policy = solve(solver, cryingbaby)
b_hungry = DiscreteBelief(cryingbaby,[.1,.9])
println(action(ghs_policy, b_hungry))
