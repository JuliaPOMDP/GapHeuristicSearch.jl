using POMDPs
using POMDPPolicies
using POMDPSimulators
using POMDPModels
using BeliefUpdaters

using GapHeuristicSearch

cryingbaby = BabyPOMDP()
solver = GapHeuristicSearchSolver(DiscreteUpdater(cryingbaby),
                                Rmax=0.0,  # best possible reward is baby not hungry, didnt feed
                                π=RandomPolicy(cryingbaby))
ghs_policy = solve(solver, cryingbaby)
b_hungry = DiscreteBelief(cryingbaby,[.1,.9])

println("action: ",action(ghs_policy, b_hungry))

# Use POMDPLinter to view requirements for `solve` and `action`
show_requirements(get_requirements(solve, (solver, cryingbaby)))
show_requirements(get_requirements(action, (ghs_policy, b_hungry)))
